from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Sequence

from .ai_processor import AIProcessingError, AIProcessor
from .cleaner import clean_articles
from .config import AppConfig, load_config
from .email_sender import send_email, send_html_file
from .i18n import Locale
from .models import ProcessedResult
from .pipeline_report import (
    PipelineReport,
    PipelineRunError,
    attach_report_warnings,
    build_failure_result,
    unwrap_pipeline_error,
)
from .rss_fetcher import fetch_all_rss
from .utils import PipelineTimer, file_lock, setup_logging


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News digest generation and delivery")
    parser.add_argument("--config", default="config.yaml", help="Config file path (default: config.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Output HTML only, do not send email")
    parser.add_argument(
        "--ai-debug",
        action="store_true",
        help="Log full AI requests/responses to a debug file",
    )
    parser.add_argument(
        "--ai-debug-dir",
        default="",
        help="AI debug log directory (default: config.yaml ai.debug_dump_dir)",
    )
    parser.add_argument(
        "--log-level",
        default="",
        help="Log level (DEBUG/INFO/WARNING/ERROR)",
    )
    parser.add_argument(
        "--send-html",
        default="",
        metavar="FILE",
        help="Skip fetch and AI, send the specified HTML file as email",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Log elapsed time for each pipeline stage",
    )
    return parser.parse_args(argv)


async def run_once(
    cfg: AppConfig,
    logger: logging.Logger,
    dry_run: bool,
    ai_debug: bool,
    locale: Locale,
    timer: PipelineTimer,
) -> ProcessedResult:
    report = PipelineReport()
    try:
        logger.info("Pipeline started")

        async with timer.async_stage("RSS fetch"):
            fetch_result = await fetch_all_rss(
                cfg.rss_sources,
                logger,
                missing_pub_date_strict=cfg.filter.rss_missing_pub_date_strict,
                timer=timer,
            )
        report.extend_warnings(fetch_result.warnings)

        with timer.stage("Clean"):
            cleaned_articles = clean_articles(
                fetch_result.articles,
                hours_back=cfg.filter.hours_back,
                max_content_length=cfg.filter.max_content_length,
                timezone_name=cfg.schedule.timezone,
                logger=logger,
                timer=timer,
            )

        if not cleaned_articles:
            logger.warning("No articles after cleaning, sending empty digest")
            empty_text = str(locale.fallback_texts.get("no_articles", "") or "").strip() or "No qualifying news today."
            result = ProcessedResult(
                articles=[],
                categories=[],
                summary_lines=[empty_text],
                degraded=False,
                warnings=[],
            )
        else:
            ai = AIProcessor(
                cfg.ai, cfg.env, logger,
                debug_capture_all=ai_debug, locale=locale, timer=timer,
            )
            try:
                async with timer.async_stage("AI process"):
                    result = await ai.process_articles(cleaned_articles)
            except AIProcessingError as exc:
                logger.error("AI processing failed: %s", exc)
                if cfg.ai.fallback_send_raw_email:
                    logger.warning("Triggering degraded email mode")
                    warning_text = (
                        str(locale.fallback_texts.get("warning", "") or "").strip()
                        or cfg.ai.fallback_warning_text
                    )
                    result = ai.build_degraded_result(
                        cleaned_articles,
                        warning_text,
                    )
                else:
                    raise
            finally:
                await ai.aclose()

        attach_report_warnings(
            result,
            report.warnings,
            include_warnings=cfg.failure_delivery.enabled and cfg.failure_delivery.include_warnings,
            locale=locale,
        )

        with timer.stage("Email"):
            send_email(
                result=result,
                email_cfg=cfg.email,
                env_cfg=cfg.env,
                logger=logger,
                dry_run=dry_run,
                timezone_name=cfg.schedule.timezone,
                locale=locale,
                timer=timer,
            )

        logger.info(
            "Pipeline complete: articles=%d, categories=%d, summary_lines=%d, degraded=%s, warnings=%d",
            len(result.articles),
            len(result.categories),
            len(result.summary_lines),
            result.degraded,
            len(result.warnings),
        )
        timer.summary(logger)
        return result
    except Exception as exc:
        raise PipelineRunError(exc, report.warnings) from exc


def send_failure_email(
    *,
    exc: BaseException,
    cfg: AppConfig,
    logger: logging.Logger,
    dry_run: bool,
    locale: Locale,
    timer: PipelineTimer,
) -> None:
    cause, warnings = unwrap_pipeline_error(exc)
    result = build_failure_result(cause, warnings=warnings, locale=locale)
    with timer.stage("Failure email"):
        send_email(
            result=result,
            email_cfg=cfg.email,
            env_cfg=cfg.env,
            logger=logger,
            dry_run=dry_run,
            timezone_name=cfg.schedule.timezone,
            locale=locale,
            timer=timer,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        locale = Locale(cfg.locale)
    except Exception as exc:
        print(f"Config load failed: {exc}", file=sys.stderr)
        return 1

    if args.log_level:
        cfg.logging.level = args.log_level.upper()
    if args.ai_debug_dir:
        cfg.ai.debug_dump_dir = args.ai_debug_dir

    logger = setup_logging(
        level=cfg.logging.level,
        file_path=cfg.logging.file,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
    )

    if args.send_html:
        try:
            send_html_file(
                html_path=args.send_html,
                email_cfg=cfg.email,
                env_cfg=cfg.env,
                logger=logger,
                locale=locale,
            )
        except Exception:
            logger.exception("HTML email sending failed")
            return 1
        return 0

    lock_path = "logs/.news-digest.lock"
    timer = PipelineTimer(enabled=args.timing)
    try:
        with file_lock(lock_path):
            asyncio.run(
                run_once(
                    cfg,
                    logger,
                    dry_run=args.dry_run,
                    ai_debug=args.ai_debug,
                    locale=locale,
                    timer=timer,
                )
            )
    except FileExistsError:
        logger.error("Another instance is already running, exiting: %s", lock_path)
        return 1
    except Exception as exc:
        logger.exception("Pipeline exited with error")
        if cfg.failure_delivery.enabled:
            try:
                send_failure_email(
                    exc=exc,
                    cfg=cfg,
                    logger=logger,
                    dry_run=args.dry_run,
                    locale=locale,
                    timer=timer,
                )
            except Exception:
                logger.exception("Failure email sending failed")
        timer.summary(logger)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
