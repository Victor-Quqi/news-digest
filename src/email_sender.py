from __future__ import annotations

import logging
import re
import smtplib
from collections import defaultdict
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import EmailConfig, EnvConfig
from .i18n import Locale
from .models import ProcessedArticle, ProcessedResult
from .utils import PipelineTimer, format_pub_datetime, today_str


_PREHEADER_MAX_LEN = 140


def _build_jinja_env(locale: Locale) -> Environment:
    template_dir = Path(__file__).resolve().parent / "templates"
    template_path = str(template_dir)
    if template_path.startswith("\\\\?\\"):
        template_path = template_path[4:]
    env = Environment(
        loader=FileSystemLoader(template_path),
        autoescape=select_autoescape(["html", "xml"]),
        extensions=["jinja2.ext.i18n"],
        trim_blocks=True,
        lstrip_blocks=True,
    )
    locale.install_jinja2(env)
    return env


def _group_articles(
    articles: List[ProcessedArticle],
    categories: List[str],
    locale: Locale,
) -> List[Dict[str, object]]:
    if not categories:
        ordered_articles = sorted(articles, key=lambda x: x.pub_date, reverse=True)
        return [{"name": "", "articles": ordered_articles, "show_heading": False}]

    grouped: Dict[str, List[ProcessedArticle]] = defaultdict(list)
    for item in articles:
        category = (item.category or "").strip()
        if not category:
            continue
        grouped[category].append(item)

    for key in grouped:
        grouped[key].sort(key=lambda x: x.pub_date, reverse=True)

    ordered_categories = [c for c in categories if c in grouped]
    for key in grouped:
        if key not in ordered_categories:
            ordered_categories.append(key)

    return [{"name": cat, "articles": grouped[cat], "show_heading": True} for cat in ordered_categories]


def _linkify_summary_line(line: str, id_to_link: Dict[int, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        ref_id = int(match.group(1))
        link = id_to_link.get(ref_id)
        if not link:
            return match.group(0)
        return (
            f'<a href="{link}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#3b82f6;text-decoration:none;">[{ref_id}]</a>'
        )

    return re.sub(r"\[(\d+)\]", repl, line)


def _normalize_preheader_text(text: str) -> str:
    cleaned = re.sub(r"\[(\d+)\]", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,.;:-")


def _truncate_preheader_text(parts: List[str], max_len: int = _PREHEADER_MAX_LEN) -> str:
    result = ""
    for part in parts:
        normalized = _normalize_preheader_text(part)
        if not normalized:
            continue
        candidate = normalized if not result else f"{result}; {normalized}"
        if len(candidate) <= max_len:
            result = candidate
            continue
        if result:
            return result
        return normalized[: max_len - 1].rstrip() + "..."
    return result


def render_email_html(
    result: ProcessedResult,
    date_text: str,
    timezone_name: str = "Asia/Shanghai",
    *,
    locale: Locale,
    template_name: str = "email.html",
) -> str:
    env = _build_jinja_env(locale=locale)
    template = env.get_template(template_name)

    id_to_link = {item.id: item.link for item in result.articles}
    summary_lines_html = [_linkify_summary_line(line, id_to_link) for line in result.summary_lines]
    grouped = _group_articles(result.articles, result.categories, locale=locale)
    preheader_text = _truncate_preheader_text(result.summary_lines)

    return template.render(
        date_text=date_text,
        count=len(result.articles),
        preheader_text=preheader_text,
        summary_lines_html=summary_lines_html,
        grouped=grouped,
        degraded=result.degraded,
        warnings=result.warnings,
        timezone_name=timezone_name,
        format_pub_datetime=format_pub_datetime,
        locale=locale,
        html_lang=locale.lang,
    )


def _build_html_message(
    *,
    subject: str,
    html: str,
    env_cfg: EnvConfig,
    recipients: List[str],
) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = env_cfg.smtp_user
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html", "utf-8"))
    return msg


def send_email(
    result: ProcessedResult,
    email_cfg: EmailConfig,
    env_cfg: EnvConfig,
    logger: logging.Logger,
    dry_run: bool = False,
    timezone_name: str = "Asia/Shanghai",
    output_dir: str = "logs",
    *,
    locale: Locale,
    template_name: str = "email.html",
    timer: PipelineTimer | None = None,
) -> str:
    if timer is None:
        timer = PipelineTimer(enabled=False)

    with timer.stage("  Render"):
        date_text = today_str(timezone_name)
        html = render_email_html(
            result=result,
            date_text=date_text,
            timezone_name=timezone_name,
            locale=locale,
            template_name=template_name,
        )
        subject = email_cfg.subject or locale.email_subject or "News Digest"
        if result.degraded:
            degraded_label = locale.t("AI Degraded Mode")
            subject = f"{subject} [{degraded_label}]"

    with timer.stage("  Send"):
        if dry_run:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = Path(output_dir) / f"news-digest-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
            output_path.write_text(html, encoding="utf-8")
            logger.info("Dry-run mode: email HTML saved to %s", output_path)
            return str(output_path)

        msg = _build_html_message(
            subject=subject,
            html=html,
            env_cfg=env_cfg,
            recipients=email_cfg.recipients,
        )
        _smtp_send(env_cfg, email_cfg.recipients, msg)
        logger.info("Email sent successfully: recipients=%d", len(email_cfg.recipients))
        return ""


def send_html_file(
    html_path: str,
    email_cfg: EmailConfig,
    env_cfg: EnvConfig,
    logger: logging.Logger,
    *,
    locale: Locale,
    subject: str = "",
) -> None:
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    html = path.read_text(encoding="utf-8")
    subject = subject or email_cfg.subject or locale.email_subject or "News Digest"

    msg = _build_html_message(
        subject=subject,
        html=html,
        env_cfg=env_cfg,
        recipients=email_cfg.recipients,
    )
    _smtp_send(env_cfg, email_cfg.recipients, msg)
    logger.info("HTML email sent successfully: file=%s, recipients=%d", html_path, len(email_cfg.recipients))


def _smtp_send(env_cfg: EnvConfig, recipients: List[str], msg: MIMEMultipart) -> None:
    # Port 465 uses implicit SSL (SMTP_SSL); port 587 uses STARTTLS (SMTP + starttls()).
    # Mixing them up causes a connection timeout because the handshake methods differ.
    if env_cfg.smtp_port == 465:
        with smtplib.SMTP_SSL(env_cfg.smtp_host, env_cfg.smtp_port, timeout=30) as server:
            server.login(env_cfg.smtp_user, env_cfg.smtp_password)
            server.sendmail(env_cfg.smtp_user, recipients, msg.as_string())
    else:
        with smtplib.SMTP(env_cfg.smtp_host, env_cfg.smtp_port, timeout=30) as server:
            server.starttls()
            server.login(env_cfg.smtp_user, env_cfg.smtp_password)
            server.sendmail(env_cfg.smtp_user, recipients, msg.as_string())
