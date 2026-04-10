from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
import unicodedata
from typing import Any, Awaitable, Callable, Dict, List, Sequence, Tuple, TypeVar

import tiktoken

from .ai_debug import AIDebugSink
from .ai_transport import AITransport
from .ai_processor_types import AIProcessingError
from .config import AIConfig, AIRetryTarget, EnvConfig
from .i18n import Locale
from .models import CleanedArticle, ProcessedArticle, ProcessedResult
from .utils import PipelineTimer, json_dumps


_T = TypeVar("_T")


class AIProcessor:
    def __init__(
        self,
        ai_config: AIConfig,
        env_config: EnvConfig,
        logger: logging.Logger,
        debug_capture_all: bool = False,
        *,
        locale: Locale,
        timer: PipelineTimer | None = None,
    ):
        self.cfg = ai_config
        self.logger = logger
        self.locale = locale
        self._timer = timer or PipelineTimer(enabled=False)
        self._debug = AIDebugSink(ai_config, logger, capture_all=debug_capture_all)
        self._transport = AITransport(ai_config, env_config, logger, self._debug)

        token_model = self._targets_for_step("summarization")[0].model
        self.encoding = None
        try:
            self.encoding = tiktoken.encoding_for_model(token_model)
        except Exception:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as exc:
                self.encoding = None
                self.logger.warning("tiktoken init failed, falling back to char-based token estimation: %s", exc)

        self._semaphore = asyncio.Semaphore(4)

        self._one_line_hard_units = max(float(self.cfg.one_line_hard_units), 1.0)
        self._one_line_soft_units = max(
            float(self.cfg.one_line_soft_units), self._one_line_hard_units
        )
        self._one_line_trim_target_units = min(
            max(float(self.cfg.one_line_trim_target_units), 1.0), self._one_line_soft_units
        )

        self._summary_line_target_len = max(int(self.cfg.summary_line_target_len), 1)
        self._summary_line_hard_limit = max(
            float(self.cfg.summary_line_hard_len), float(self._summary_line_target_len)
        )
        self._summary_line_soft_limit = max(
            float(self.cfg.summary_line_soft_len), self._summary_line_hard_limit
        )
        self.logger.debug("AI HTTP proxy from env: %s", env_config.openai_use_env_proxy)

    async def process_articles(self, articles: List[CleanedArticle]) -> ProcessedResult:
        if not articles:
            return ProcessedResult(
                articles=[],
                categories=[],
                summary_lines=[self._locale_fallback_text("no_articles", "No qualifying news today.")],
            )

        with self._timer.stage("  Token estimate"):
            threshold = int(self.cfg.context_window * self.cfg.shard_threshold_ratio)
            estimated_tokens = self._estimate_tokens(articles)
        self.logger.info(
            "AI sharding decision: articles=%d, estimated_tokens=%d, threshold=%d",
            len(articles),
            estimated_tokens,
            threshold,
        )

        summary_task = asyncio.create_task(
            self._run_summarization(articles, estimated_tokens=estimated_tokens, threshold=threshold)
        )
        categorization_task = asyncio.create_task(
            self._run_categorization(articles, threshold=threshold)
        )

        try:
            summary_map = await summary_task
        except Exception:
            await self._cancel_task(categorization_task)
            raise

        with self._timer.stage("  Build articles"):
            processed_articles = self._build_processed_articles(articles, summary_map)

        try:
            summary_lines = await self._run_overview(processed_articles)
        except Exception:
            await self._cancel_task(categorization_task)
            raise

        categories: List[str] = []
        try:
            category_map = await categorization_task
        except AIProcessingError as exc:
            if self._categorization_is_strict():
                raise AIProcessingError(f"Categorization failed (strict mode): {exc}") from exc
            self.logger.warning(
                "Categorization failed, continuing without category grouping: %s",
                exc,
            )
            category_map = None

        if category_map:
            with self._timer.stage("  Categories"):
                self._apply_category_map(processed_articles, category_map)
                categories = self._collect_categories(processed_articles)

        return ProcessedResult(
            articles=processed_articles,
            categories=categories,
            summary_lines=summary_lines,
            degraded=False,
            warnings=[],
        )

    def build_degraded_result(
        self, articles: List[CleanedArticle], warning_text: str
    ) -> ProcessedResult:
        resolved_warning_text = str(warning_text or "").strip() or self._locale_fallback_text(
            "warning",
            "",
        )
        degraded_articles = [
            ProcessedArticle(
                id=item.id,
                title=item.title,
                link=item.link,
                pub_date=item.pub_date,
                source=item.source,
                one_line="",
                category="",
            )
            for item in articles
        ]
        return ProcessedResult(
            articles=degraded_articles,
            categories=[],
            summary_lines=[self._locale_fallback_text("overview_failed", "AI overview generation failed. Original article list shown below.")],
            degraded=True,
            warnings=[resolved_warning_text] if resolved_warning_text else [],
        )

    def _locale_fallback_text(self, key: str, default: str) -> str:
        text = str(self.locale.fallback_texts.get(key, "") or "").strip()
        return text or default

    def _render_prompt_with_default(
        self,
        *,
        step: str,
        role: str,
        default: str,
        **kwargs: Any,
    ) -> str:
        template = self.locale.get(f"prompts.{step}.{role}", default)
        return str(template).format(**kwargs)

    async def aclose(self) -> None:
        await self._transport.aclose()

    def _dump_debug(self, event: str, payload: Dict[str, Any], force: bool = False) -> None:
        self._debug.dump(event, payload, force=force)

    async def _cancel_task(self, task: asyncio.Task[Any]) -> None:
        if task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _run_json_phase(
        self,
        *,
        step: str,
        log_label: str,
        request_fn: Callable[[AIRetryTarget], Awaitable[Any]],
        validate_fn: Callable[[Any], _T],
        timer_api_label: str,
        timer_validate_label: str,
    ) -> _T:
        targets = self._targets_for_step(step)
        threshold = max(int(self.cfg.retry_target_failure_threshold), 1)
        schema_attempt = 0
        call_attempt = 0
        candidate_index = 0
        candidate_failures = 0
        api_time = 0.0
        val_time = 0.0

        try:
            while True:
                data: object = None
                target = targets[candidate_index]
                t0 = time.monotonic()
                try:
                    data = await request_fn(target)
                    api_time += time.monotonic() - t0

                    t0 = time.monotonic()
                    result = validate_fn(data)
                    val_time += time.monotonic() - t0
                    return result
                except (json.JSONDecodeError, ValueError) as exc:
                    val_time += time.monotonic() - t0
                    failure_count = candidate_failures + 1
                    self._dump_debug(
                        f"{step}.schema_or_validation_error",
                        {
                            "attempt": schema_attempt,
                            "target": self._target_payload(target),
                            "candidate_failures": failure_count,
                            "error": str(exc),
                            "response_data": data,
                        },
                        force=True,
                    )
                    next_attempt = schema_attempt + 1
                    if next_attempt > self.cfg.schema_retry_max:
                        raise AIProcessingError(f"{step} schema/validation failed: {exc}") from exc
                    schema_attempt = next_attempt
                    candidate_index, candidate_failures, switched = self._advance_retry_target_state(
                        targets=targets,
                        candidate_index=candidate_index,
                        candidate_failures=candidate_failures,
                    )
                    self._log_retry(
                        log_label=log_label,
                        step=step,
                        failure_kind="schema_or_validation",
                        error=exc,
                        target=target,
                        candidate_failures=failure_count,
                        candidate_threshold=threshold,
                        attempt=schema_attempt,
                        budget=self.cfg.schema_retry_max,
                        switched=switched,
                        next_target=targets[candidate_index],
                    )
                    continue
                except Exception as exc:
                    failure_count = candidate_failures + 1
                    self._dump_debug(
                        f"{step}.call_error",
                        {
                            "target": self._target_payload(target),
                            "candidate_failures": failure_count,
                            "error": str(exc),
                            "response_data": data,
                        },
                        force=True,
                    )
                    if not self._should_retry_call_error(exc):
                        raise AIProcessingError(f"{step} call failed: {exc}") from exc
                    next_attempt = call_attempt + 1
                    if next_attempt > self.cfg.transient_retry_max:
                        raise AIProcessingError(f"{step} call failed: {exc}") from exc
                    call_attempt = next_attempt
                    delay = self._backoff_delay(call_attempt - 1) if self._is_transient_error(exc) else 0.0
                    candidate_index, candidate_failures, switched = self._advance_retry_target_state(
                        targets=targets,
                        candidate_index=candidate_index,
                        candidate_failures=candidate_failures,
                    )
                    self._log_retry(
                        log_label=log_label,
                        step=step,
                        failure_kind="call",
                        error=exc,
                        target=target,
                        candidate_failures=failure_count,
                        candidate_threshold=threshold,
                        attempt=call_attempt,
                        budget=self.cfg.transient_retry_max,
                        switched=switched,
                        next_target=targets[candidate_index],
                        delay=delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
        finally:
            self._timer.record(timer_api_label, api_time)
            self._timer.record(timer_validate_label, val_time)

    def _targets_for_step(self, step: str) -> List[AIRetryTarget]:
        if step in {"summarization", "categorization", "category_suggestion"}:
            return list(self.cfg.summarization_retry_targets)
        if step in {"overview", "overview_headline", "overview_groups"}:
            return list(self.cfg.overview_retry_targets)
        raise ValueError(f"Unsupported AI step: {step}")

    def _advance_retry_target_state(
        self,
        *,
        targets: Sequence[AIRetryTarget],
        candidate_index: int,
        candidate_failures: int,
    ) -> Tuple[int, int, bool]:
        threshold = max(int(self.cfg.retry_target_failure_threshold), 1)
        updated_failures = candidate_failures + 1
        if updated_failures >= threshold and candidate_index < len(targets) - 1:
            return candidate_index + 1, 0, True
        return candidate_index, updated_failures, False

    def _target_payload(self, target: AIRetryTarget) -> Dict[str, str]:
        return {
            "name": target.name,
            "base_url": target.base_url,
            "model": target.model,
        }

    def _log_retry(
        self,
        *,
        log_label: str,
        step: str,
        failure_kind: str,
        error: Exception,
        target: AIRetryTarget,
        candidate_failures: int,
        candidate_threshold: int,
        attempt: int,
        budget: int,
        switched: bool,
        next_target: AIRetryTarget,
        delay: float = 0.0,
    ) -> None:
        self.logger.warning(
            "%s retry scheduled: step=%s, failure_kind=%s, attempt=%d/%d, "
            "target=%s, model=%s, base_url=%s, candidate_failures=%d/%d, "
            "next_target=%s, next_model=%s, next_base_url=%s, switched=%s, delay=%.2fs, error=%s",
            log_label,
            step,
            failure_kind,
            attempt,
            budget,
            target.name,
            target.model,
            target.base_url,
            candidate_failures,
            candidate_threshold,
            next_target.name,
            next_target.model,
            next_target.base_url,
            switched,
            delay,
            error,
        )

    def _apply_text_length_policy(
        self,
        text: str,
        *,
        label: str,
        value_fn: Callable[[str], float],
        trim_fn: Callable[[str, float], str],
        hard_limit: float,
        soft_limit: float,
        trim_target: float,
        item_id: int | None = None,
    ) -> str:
        value = float(value_fn(text))
        context = f"id={item_id} {label}" if item_id is not None else label
        if value > soft_limit:
            shortened = trim_fn(text, trim_target)
            self.logger.warning(
                "%s too long (%.1f), trimmed to %.1f",
                context,
                value,
                float(value_fn(shortened)),
            )
            return shortened
        if value > hard_limit:
            self.logger.info("%s slightly over limit (%.1f), keeping original", context, value)
        return text

    def _estimate_tokens(self, articles: Sequence[CleanedArticle]) -> int:
        payload = [
            {
                "id": a.id,
                "title": a.title,
                "content": a.content,
            }
            for a in articles
        ]
        text = json_dumps(payload)
        if self.encoding is None:
            estimated = int(len(text) * 1.2)
            return estimated + 2000 + self.cfg.max_tokens
        return len(self.encoding.encode(text)) + 2000 + self.cfg.max_tokens

    def _build_shards(self, articles: Sequence[CleanedArticle]) -> List[List[CleanedArticle]]:
        shards: List[List[CleanedArticle]] = []
        current: List[CleanedArticle] = []
        current_chars = 0

        for article in articles:
            article_chars = len(article.title) + len(article.content)
            reaches_limit = (
                len(current) >= self.cfg.shard_max_articles
                or current_chars + article_chars > self.cfg.shard_max_chars
            )
            if current and reaches_limit:
                shards.append(current)
                current = []
                current_chars = 0

            current.append(article)
            current_chars += article_chars

        if current:
            shards.append(current)
        return shards

    async def _run_summarization(
        self,
        articles: Sequence[CleanedArticle],
        *,
        estimated_tokens: int,
        threshold: int,
    ) -> Dict[int, Dict[str, str]]:
        if estimated_tokens < threshold:
            async with self._timer.async_stage("  Summarization"):
                return await self._summarize_batch(list(articles), shard_label="Shard 1")

        with self._timer.stage("  Summary sharding"):
            shards = self._build_shards(articles)
        self.logger.info("Summarization shard mode: shard_count=%d", len(shards))
        async with self._timer.async_stage("  Summarization"):
            return await self._summarize_shards(shards)

    async def _run_categorization(
        self,
        articles: Sequence[CleanedArticle],
        *,
        threshold: int,
    ) -> Dict[int, str]:
        category_candidates = self._configured_preferred_categories()
        if not category_candidates:
            try:
                category_candidates = await self._run_category_suggestion(
                    articles,
                    threshold=threshold,
                )
            except AIProcessingError as exc:
                self.logger.warning(
                    "Category suggestion failed, continuing without suggested categories: %s",
                    exc,
                )
                category_candidates = []

        estimated_tokens = self._estimate_tokens_for_categorization(articles)
        self.logger.info(
            "Categorization sharding decision: articles=%d, estimated_tokens=%d, threshold=%d",
            len(articles),
            estimated_tokens,
            threshold,
        )

        if estimated_tokens < threshold:
            async with self._timer.async_stage("  Categorization"):
                return await self._categorize_batch(
                    list(articles),
                    shard_label="Shard 1",
                    category_candidates=category_candidates,
                    used_categories=[],
                    total_article_count=len(articles),
                )

        with self._timer.stage("  Category sharding"):
            shards = self._build_categorization_shards(articles)
        self.logger.info("Categorization shard mode: shard_count=%d", len(shards))

        merged: Dict[int, str] = {}
        used_categories: List[str] = []
        async with self._timer.async_stage("  Categorization"):
            for index, shard in enumerate(shards, start=1):
                shard_map = await self._categorize_batch(
                    list(shard),
                    shard_label=f"Shard {index}",
                    category_candidates=category_candidates,
                    used_categories=used_categories,
                    total_article_count=len(articles),
                )
                merged.update(shard_map)
                for category in shard_map.values():
                    if category not in used_categories:
                        used_categories.append(category)
        return merged

    def _configured_preferred_categories(self) -> List[str]:
        return [
            str(item).strip()
            for item in getattr(self.cfg, "preferred_categories", [])
            if str(item).strip()
        ]

    async def _run_category_suggestion(
        self,
        articles: Sequence[CleanedArticle],
        *,
        threshold: int,
    ) -> List[str]:
        estimated_tokens = self._estimate_tokens_for_category_suggestion(articles)
        max_categories = self._categorization_max_categories(len(articles))
        self.logger.info(
            "Category suggestion decision: articles=%d, estimated_tokens=%d, threshold=%d",
            len(articles),
            estimated_tokens,
            threshold,
        )

        if estimated_tokens < threshold:
            async with self._timer.async_stage("  Category suggestion"):
                return await self._suggest_categories_batch(
                    list(articles),
                    shard_label="Shard 1",
                    existing_categories=[],
                    total_article_count=len(articles),
                )

        with self._timer.stage("  Category suggestion sharding"):
            shards = self._build_category_suggestion_shards(articles)
        self.logger.info("Category suggestion shard mode: shard_count=%d", len(shards))

        merged: List[str] = []
        async with self._timer.async_stage("  Category suggestion"):
            for index, shard in enumerate(shards, start=1):
                shard_categories = await self._suggest_categories_batch(
                    list(shard),
                    shard_label=f"Shard {index}",
                    existing_categories=merged,
                    total_article_count=len(articles),
                )
                for category in shard_categories:
                    if category not in merged:
                        merged.append(category)
                if len(merged) > max_categories:
                    merged = merged[:max_categories]
        return merged

    async def _run_overview(self, articles: List[ProcessedArticle]) -> List[str]:
        try:
            async with self._timer.async_stage("  Overview"):
                return await self._generate_overview(articles)
        except AIProcessingError as exc:
            if self.cfg.overview_text_fallback:
                self.logger.warning("Overview JSON generation failed, trying text fallback: %s", exc)
                try:
                    async with self._timer.async_stage("  Overview text fallback"):
                        return await self._generate_overview_text_fallback(articles)
                except AIProcessingError as fallback_exc:
                    if self.cfg.overview_local_fallback:
                        self.logger.warning(
                            "Overview text fallback failed, using local rule-based overview: %s",
                            fallback_exc,
                        )
                        with self._timer.stage("  Overview local fallback"):
                            return self._build_local_overview(articles, locale=self.locale)
                    raise AIProcessingError(
                        f"Overview text fallback failed (strict mode, local fallback disabled): {fallback_exc}"
                    ) from fallback_exc
            if self.cfg.overview_local_fallback:
                self.logger.warning("Overview generation failed, using local rule-based overview: %s", exc)
                with self._timer.stage("  Overview local fallback"):
                    return self._build_local_overview(articles, locale=self.locale)
            raise AIProcessingError(f"Overview generation failed (strict mode): {exc}") from exc

    def _estimate_tokens_for_categorization(
        self,
        articles: Sequence[CleanedArticle],
    ) -> int:
        payload = [
            {
                "id": article.id,
                "title": article.title,
                "content": self._categorization_content(article.content),
            }
            for article in articles
        ]
        text = json_dumps(payload)
        if self.encoding is None:
            estimated = int(len(text) * 1.2)
            return estimated + 2000 + self.cfg.max_tokens
        return len(self.encoding.encode(text)) + 2000 + self.cfg.max_tokens

    def _estimate_tokens_for_category_suggestion(
        self,
        articles: Sequence[CleanedArticle],
    ) -> int:
        payload = [
            {
                "title": article.title,
                "content": self._category_suggestion_content(article.content),
            }
            for article in articles
        ]
        text = json_dumps(payload)
        if self.encoding is None:
            estimated = int(len(text) * 1.2)
            return estimated + 2000 + self.cfg.max_tokens
        return len(self.encoding.encode(text)) + 2000 + self.cfg.max_tokens

    def _build_categorization_shards(
        self,
        articles: Sequence[CleanedArticle],
    ) -> List[List[CleanedArticle]]:
        shards: List[List[CleanedArticle]] = []
        current: List[CleanedArticle] = []
        current_chars = 0

        for article in articles:
            article_chars = len(article.title) + len(self._categorization_content(article.content))
            reaches_limit = (
                len(current) >= self.cfg.shard_max_articles
                or current_chars + article_chars > self.cfg.shard_max_chars
            )
            if current and reaches_limit:
                shards.append(current)
                current = []
                current_chars = 0

            current.append(article)
            current_chars += article_chars

        if current:
            shards.append(current)
        return shards

    def _build_category_suggestion_shards(
        self,
        articles: Sequence[CleanedArticle],
    ) -> List[List[CleanedArticle]]:
        shards: List[List[CleanedArticle]] = []
        current: List[CleanedArticle] = []
        current_chars = 0

        for article in articles:
            article_chars = len(article.title) + len(self._category_suggestion_content(article.content))
            reaches_limit = (
                len(current) >= self.cfg.shard_max_articles
                or current_chars + article_chars > self.cfg.shard_max_chars
            )
            if current and reaches_limit:
                shards.append(current)
                current = []
                current_chars = 0

            current.append(article)
            current_chars += article_chars

        if current:
            shards.append(current)
        return shards

    def _categorization_content(self, content: str) -> str:
        return self._truncate_prompt_content(content, max_chars=1200)

    def _category_suggestion_content(self, content: str) -> str:
        return self._truncate_prompt_content(content, max_chars=400)

    def _truncate_prompt_content(self, text: str, *, max_chars: int) -> str:
        content = str(text or "").strip()
        if max_chars <= 0 or len(content) <= max_chars:
            return content
        head_chars = max(max_chars // 2, 1)
        tail_chars = max(max_chars - head_chars - 1, 0)
        if tail_chars <= 0:
            return content[:max_chars]
        return f"{content[:head_chars].rstrip()}…{content[-tail_chars:].lstrip()}"

    def _categorization_is_strict(self) -> bool:
        return bool(getattr(self.cfg, "categorization_strict", True))

    def _categorization_max_categories(self, article_count: int) -> int:
        if article_count <= 0:
            return 1
        dynamic_cap = round(1.5 * math.log2(max(article_count, 1)))
        bounded_cap = max(4, min(dynamic_cap, 10))
        return min(article_count, bounded_cap)

    def _categorization_language_instruction(self, locale: Locale) -> str:
        if locale.lang.lower().startswith("zh"):
            return "分类名尽量用中文；AI、IPO、ETF 等常见缩写可保留。"
        if locale.lang.lower().startswith("en"):
            return "Category labels must be written in English."
        return ""

    def _category_suggestion_policy_instruction(
        self,
        *,
        locale: Locale,
        max_categories: int,
        has_existing_categories: bool,
    ) -> str:
        is_english = locale.lang.lower().startswith("en")
        if is_english:
            lines = [
                "Categories must be short and generic, like section names.",
                "Usually keep labels to 1-3 words.",
                f"Keep the total category count under {max_categories} when possible.",
            ]
            if has_existing_categories:
                lines.insert(0, "If an existing category still fits, reuse it.")
            else:
                lines.insert(0, "Only keep the main recurring themes of this batch.")
            return "\n".join(lines)

        lines = [
            "分类名必须简短、通用，像栏目名。",
            "最好 2~4 字，必要时可稍长。",
            f"本批分类总数尽量少于 {max_categories} 个。",
        ]
        if has_existing_categories:
            lines.insert(0, "如果已有分类仍然贴切，优先复用。")
        else:
            lines.insert(0, "只保留这一批新闻里反复出现的主线。")
        return "\n".join(lines)

    def _categorization_policy_instruction(
        self,
        *,
        locale: Locale,
        max_categories: int,
        category_candidates_name: str,
        has_category_candidates: bool,
        has_used_categories: bool,
    ) -> str:
        is_english = locale.lang.lower().startswith("en")
        if has_category_candidates:
            if is_english:
                lines = [
                    f"Prefer using labels directly from {category_candidates_name}.",
                ]
                if has_used_categories:
                    lines.insert(
                        1,
                        "If an already used category still fits, reuse it.",
                    )
                lines.extend(
                    [
                        f"Only create a new category when none of the labels in {category_candidates_name} fit.",
                        "A new category should be short and generic, like a section name.",
                        "Usually keep it to 1-3 words.",
                        f"Keep the total category count under {max_categories} when possible.",
                    ]
                )
            else:
                lines = [
                    f"优先直接使用 {category_candidates_name} 里的分类名。",
                ]
                if has_used_categories:
                    lines.insert(
                        1,
                        "如果已有分类仍然贴切，优先复用已有分类。",
                    )
                lines.extend(
                    [
                        f"只有当 {category_candidates_name} 都不贴切时，才允许新建分类。",
                        "新建分类也要简短、通用，像栏目名。",
                        "最好 2~4 字，必要时可稍长。",
                        f"本批分类总数尽量少于 {max_categories} 个。",
                    ]
                )
            return "\n".join(lines)

        if is_english:
            lines = [
                "Categories must be short and generic, like section names.",
                "Reuse an existing category when it still fits. Only create a new one when the existing categories clearly do not fit.",
                "Usually keep labels to 1-3 words.",
                f"Keep the total category count under {max_categories} when possible.",
            ]
        else:
            lines = [
                "分类名必须简短、通用，像栏目名。",
                "如果已有分类仍然贴切，优先复用；只有明显放不进去时才新建。",
                "最好 2~4 字，必要时可稍长。",
                f"本批分类总数尽量少于 {max_categories} 个。",
            ]
        return "\n".join(lines)

    def _build_categorization_prompt_kwargs(
        self,
        *,
        locale: Locale,
        shard_article_count: int,
        total_article_count: int,
        preferred_categories: Sequence[str],
        suggested_categories: Sequence[str],
        used_categories: Sequence[str],
        example_output: str,
        articles_json: str,
    ) -> Dict[str, Any]:
        is_english = locale.lang.lower().startswith("en")
        preferred_categories = [str(item).strip() for item in preferred_categories if str(item).strip()]
        suggested_categories = [str(item).strip() for item in suggested_categories if str(item).strip()]
        used_categories = [str(item).strip() for item in used_categories if str(item).strip()]

        category_candidates_name = ""
        category_candidates: List[str] = []
        if preferred_categories:
            category_candidates_name = "preferred_categories"
            category_candidates = preferred_categories
        elif suggested_categories:
            category_candidates_name = "suggested_categories"
            category_candidates = suggested_categories

        category_candidates_block = ""
        if category_candidates:
            category_candidates_block = f"{category_candidates_name}={json_dumps(category_candidates)}\n"

        used_instruction = ""
        used_block = ""
        if used_categories:
            used_instruction = (
                "already_used_categories lists labels that earlier shards have already used; if one still fits, reuse it to keep the whole run consistent.\n"
                if is_english
                else "already_used_categories 是前面分片已经用过的分类名；如果仍然贴切，优先复用以保持整批一致。\n"
            )
            used_block = f"already_used_categories={json_dumps(used_categories)}\n"

        return {
            "count": shard_article_count,
            "max_categories": self._categorization_max_categories(total_article_count),
            "category_language_instruction": self._categorization_language_instruction(locale),
            "already_used_categories_instruction": used_instruction,
            "categorization_policy_instruction": self._categorization_policy_instruction(
                locale=locale,
                max_categories=self._categorization_max_categories(total_article_count),
                category_candidates_name=category_candidates_name,
                has_category_candidates=bool(category_candidates),
                has_used_categories=bool(used_categories),
            ),
            "category_candidates_block": category_candidates_block,
            "already_used_categories_block": used_block,
            "example_output": example_output,
            "articles_json": articles_json,
        }

    def _build_category_suggestion_prompt_kwargs(
        self,
        *,
        locale: Locale,
        shard_article_count: int,
        total_article_count: int,
        existing_categories: Sequence[str],
        example_output: str,
        articles_json: str,
    ) -> Dict[str, Any]:
        is_english = locale.lang.lower().startswith("en")
        existing_categories = [str(item).strip() for item in existing_categories if str(item).strip()]

        existing_instruction = ""
        existing_block = ""
        if existing_categories:
            existing_instruction = (
                "existing_categories lists labels that earlier shards have already produced; if one still fits, reuse it to keep the whole run consistent.\n"
                if is_english
                else "existing_categories 是前面分片已经产出的分类名；如果仍然贴切，优先复用以保持整批一致。\n"
            )
            existing_block = f"existing_categories={json_dumps(existing_categories)}\n"

        return {
            "count": shard_article_count,
            "max_categories": self._categorization_max_categories(total_article_count),
            "category_language_instruction": self._categorization_language_instruction(locale),
            "existing_categories_instruction": existing_instruction,
            "category_suggestion_policy_instruction": self._category_suggestion_policy_instruction(
                locale=locale,
                max_categories=self._categorization_max_categories(total_article_count),
                has_existing_categories=bool(existing_categories),
            ),
            "existing_categories_block": existing_block,
            "example_output": example_output,
            "articles_json": articles_json,
        }

    async def _summarize_shards(
        self, shards: Sequence[Sequence[CleanedArticle]]
    ) -> Dict[int, Dict[str, str]]:
        async def run_single(
            shard: Sequence[CleanedArticle], label: str
        ) -> Dict[int, Dict[str, str]]:
            async with self._semaphore:
                async with self._timer.async_stage(f"    {label}"):
                    return await self._summarize_batch(list(shard), shard_label=label)

        results = await asyncio.gather(
            *(run_single(s, f"Shard {i+1}") for i, s in enumerate(shards))
        )
        merged: Dict[int, Dict[str, str]] = {}
        for item in results:
            merged.update(item)
        return merged

    async def _summarize_batch(
        self, articles: List[CleanedArticle], *, shard_label: str = "",
    ) -> Dict[int, Dict[str, str]]:
        expected_ids = {a.id for a in articles}
        title_by_id = {a.id: a.title for a in articles}
        prefix = f"      {shard_label} " if shard_label else "    "
        return await self._run_json_phase(
            step="summarization",
            log_label="Summarization",
            request_fn=lambda target: self._request_summarization(articles, locale=self.locale, target=target),
            validate_fn=lambda data: self._normalize_summarization_result(
                data,
                expected_ids=expected_ids,
                title_by_id=title_by_id,
            ),
            timer_api_label=f"{prefix}API",
            timer_validate_label=f"{prefix}Validate",
        )

    async def _categorize_batch(
        self,
        articles: List[CleanedArticle],
        *,
        shard_label: str = "",
        category_candidates: Sequence[str],
        used_categories: Sequence[str],
        total_article_count: int,
    ) -> Dict[int, str]:
        expected_ids = {a.id for a in articles}
        prefix = f"      {shard_label} " if shard_label else "    "
        return await self._run_json_phase(
            step="categorization",
            log_label="Categorization",
            request_fn=lambda target: self._request_categorization(
                articles,
                locale=self.locale,
                target=target,
                category_candidates=category_candidates,
                used_categories=used_categories,
                total_article_count=total_article_count,
            ),
            validate_fn=lambda data: self._normalize_categorization_result(
                data,
                expected_ids=expected_ids,
                locale=self.locale,
            ),
            timer_api_label=f"{prefix}Category API",
            timer_validate_label=f"{prefix}Category Validate",
        )

    async def _suggest_categories_batch(
        self,
        articles: List[CleanedArticle],
        *,
        shard_label: str = "",
        existing_categories: Sequence[str],
        total_article_count: int,
    ) -> List[str]:
        prefix = f"      {shard_label} " if shard_label else "    "
        return await self._run_json_phase(
            step="category_suggestion",
            log_label="Category suggestion",
            request_fn=lambda target: self._request_category_suggestion(
                articles,
                locale=self.locale,
                target=target,
                existing_categories=existing_categories,
                total_article_count=total_article_count,
            ),
            validate_fn=lambda data: self._normalize_category_suggestion_result(
                data,
                locale=self.locale,
                max_categories=self._categorization_max_categories(total_article_count),
            ),
            timer_api_label=f"{prefix}Suggest API",
            timer_validate_label=f"{prefix}Suggest Validate",
        )

    async def _generate_overview(self, articles: List[ProcessedArticle]) -> List[str]:
        headline = await self._generate_overview_headline(articles)
        groups = await self._generate_overview_groups(articles, headline=headline)
        return self._postprocess_overview_lines(
            [headline, *groups],
            articles,
            locale=self.locale,
        )

    async def _generate_overview_headline(
        self,
        articles: List[ProcessedArticle],
    ) -> str:
        return await self._run_json_phase(
            step="overview_headline",
            log_label="Overview headline",
            request_fn=lambda target: self._request_overview_headline(
                articles,
                locale=self.locale,
                target=target,
            ),
            validate_fn=lambda data: self._validate_overview_headline(
                data,
                locale=self.locale,
            ),
            timer_api_label="    Headline API",
            timer_validate_label="    Headline Validate",
        )

    async def _generate_overview_groups(
        self,
        articles: List[ProcessedArticle],
        *,
        headline: str,
    ) -> List[str]:
        valid_ids = {a.id for a in articles}
        return await self._run_json_phase(
            step="overview_groups",
            log_label="Overview groups",
            request_fn=lambda target: self._request_overview_groups(
                articles,
                headline=headline,
                locale=self.locale,
                target=target,
            ),
            validate_fn=lambda data: self._validate_overview_groups(
                data,
                valid_ids,
                locale=self.locale,
            ),
            timer_api_label="    Groups API",
            timer_validate_label="    Groups Validate",
        )

    async def _generate_overview_text_fallback(
        self, articles: List[ProcessedArticle]
    ) -> List[str]:
        valid_ids = {a.id for a in articles}
        targets = self._targets_for_step("overview")
        threshold = max(int(self.cfg.retry_target_failure_threshold), 1)
        call_attempt = 0
        candidate_index = 0
        candidate_failures = 0

        while True:
            headline_text = ""
            groups_text = ""
            target = targets[candidate_index]
            try:
                headline_text = await self._request_overview_headline_text(
                    articles,
                    locale=self.locale,
                    target=target,
                )
                headline = self._validate_overview_headline(headline_text, locale=self.locale)

                groups_text = await self._request_overview_groups_text(
                    articles,
                    headline=headline,
                    locale=self.locale,
                    target=target,
                )
                raw_lines = self._coerce_summary_lines(groups_text)
                if not raw_lines:
                    raise ValueError("plain text overview groups are empty")

                normalized = [str(x or "").strip() for x in raw_lines if str(x or "").strip()]
                if not normalized:
                    raise ValueError("plain text overview groups are empty")

                normalized = self._sanitize_overview_lines(
                    normalized,
                    strip_first_line_refs=False,
                )
                groups = self._validate_overview_groups(
                    normalized,
                    valid_ids,
                    locale=self.locale,
                )
                return self._postprocess_overview_lines(
                    [headline, *groups],
                    articles,
                    locale=self.locale,
                )
            except Exception as exc:
                failure_count = candidate_failures + 1
                self._dump_debug(
                    "overview.text_fallback_error",
                    {
                        "target": self._target_payload(target),
                        "candidate_failures": failure_count,
                        "error": str(exc),
                        "response_text": {
                            "headline": headline_text,
                            "groups": groups_text,
                        },
                    },
                    force=True,
                )
                if not self._should_retry_call_error(exc):
                    raise AIProcessingError(f"Overview text fallback failed: {exc}") from exc
                next_attempt = call_attempt + 1
                if next_attempt > self.cfg.transient_retry_max:
                    raise AIProcessingError(f"Overview text fallback failed: {exc}") from exc
                call_attempt = next_attempt
                delay = self._backoff_delay(call_attempt - 1) if self._is_transient_error(exc) else 0.0
                candidate_index, candidate_failures, switched = self._advance_retry_target_state(
                    targets=targets,
                    candidate_index=candidate_index,
                    candidate_failures=candidate_failures,
                )
                self._log_retry(
                    log_label="Overview text fallback",
                    step="overview",
                    failure_kind="text_fallback",
                    error=exc,
                    target=target,
                    candidate_failures=failure_count,
                    candidate_threshold=threshold,
                    attempt=call_attempt,
                    budget=self.cfg.transient_retry_max,
                    switched=switched,
                    next_target=targets[candidate_index],
                    delay=delay,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                continue

    def _validate_summarization(
        self,
        data: Dict[str, Any],
        expected_ids: set[int],
        title_by_id: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        per_article = data.get("perArticle")
        if not isinstance(per_article, list):
            raise ValueError("perArticle must be an array")

        rows: List[Dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in per_article:
            if not isinstance(item, dict):
                raise ValueError("perArticle element must be an object")

            if "id" not in item or "oneLine" not in item:
                raise ValueError("perArticle element missing required fields")

            article_id = int(item["id"])
            one_line = str(item["oneLine"] or "").strip()
            if not one_line:
                raise ValueError(f"id={article_id} oneLine must not be empty")

            one_line = self._apply_text_length_policy(
                one_line,
                label="oneLine",
                value_fn=self._one_line_units,
                trim_fn=self._trim_one_line_by_units,
                hard_limit=self._one_line_hard_units,
                soft_limit=self._one_line_soft_units,
                trim_target=self._one_line_trim_target_units,
                item_id=article_id,
            )
            mismatch = self._find_summarization_alignment_mismatch(
                article_id=article_id,
                one_line=one_line,
                title_by_id=title_by_id,
            )
            if mismatch is not None:
                best_id, own_score, best_score = mismatch
                raise ValueError(
                    f"id={article_id} oneLine alignment mismatch (self={own_score:.2f}, best_id={best_id}, best={best_score:.2f})"
                )
            if article_id in seen_ids:
                raise ValueError(f"duplicate id: {article_id}")

            seen_ids.add(article_id)
            rows.append({"id": article_id, "oneLine": one_line})

        if seen_ids != expected_ids:
            missing = sorted(expected_ids - seen_ids)
            extra = sorted(seen_ids - expected_ids)
            raise ValueError(f"id set mismatch: missing={missing}, extra={extra}")

        return rows

    def _normalize_summarization_result(
        self,
        data: Dict[str, Any],
        *,
        expected_ids: set[int],
        title_by_id: Dict[int, str],
    ) -> Dict[int, Dict[str, str]]:
        parsed = self._validate_summarization(
            data,
            expected_ids,
            title_by_id,
        )
        normalized: Dict[int, Dict[str, str]] = {}
        for entry in parsed:
            normalized[entry["id"]] = {
                "one_line": entry["oneLine"],
            }
        return normalized

    def _validate_categorization(
        self,
        data: Dict[str, Any],
        expected_ids: set[int],
        *,
        locale: Locale,
    ) -> List[Dict[str, Any]]:
        per_article = data.get("perArticle")
        if not isinstance(per_article, list):
            raise ValueError("perArticle must be an array")

        rows: List[Dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in per_article:
            if not isinstance(item, dict):
                raise ValueError("perArticle element must be an object")
            if "id" not in item or "category" not in item:
                raise ValueError("perArticle element missing required fields")

            article_id = int(item["id"])
            category = self._normalize_category(str(item["category"] or ""))
            if not category:
                raise ValueError(f"id={article_id} category must not be empty")
            if article_id in seen_ids:
                raise ValueError(f"duplicate id: {article_id}")
            seen_ids.add(article_id)
            rows.append({"id": article_id, "category": category})

        if seen_ids != expected_ids:
            missing = sorted(expected_ids - seen_ids)
            extra = sorted(seen_ids - expected_ids)
            raise ValueError(f"id set mismatch: missing={missing}, extra={extra}")

        return rows

    def _normalize_categorization_result(
        self,
        data: Dict[str, Any],
        *,
        expected_ids: set[int],
        locale: Locale,
    ) -> Dict[int, str]:
        parsed = self._validate_categorization(data, expected_ids, locale=locale)
        return {entry["id"]: entry["category"] for entry in parsed}

    def _validate_category_suggestion(
        self,
        data: Dict[str, Any],
        *,
        locale: Locale,
        max_categories: int,
    ) -> List[str]:
        categories = data.get("categories")
        if not isinstance(categories, list):
            raise ValueError("categories must be an array")

        normalized: List[str] = []
        seen: set[str] = set()
        for item in categories:
            category = self._normalize_category(str(item or ""))
            if not category:
                raise ValueError("suggested category must not be empty")
            if category in seen:
                continue
            seen.add(category)
            normalized.append(category)

        if not normalized:
            raise ValueError("categories must not be empty")
        if len(normalized) > max_categories:
            raise ValueError(
                f"categories must have 1-{max_categories} items, got {len(normalized)}"
            )
        return normalized

    def _normalize_category_suggestion_result(
        self,
        data: Dict[str, Any],
        *,
        locale: Locale,
        max_categories: int,
    ) -> List[str]:
        return self._validate_category_suggestion(
            data,
            locale=locale,
            max_categories=max_categories,
        )

    def _match_tokens(self, text: str) -> set[str]:
        stopwords = {
            "今日",
            "中国",
            "美国",
            "全球",
            "市场",
            "公司",
            "发布",
            "宣布",
            "表示",
            "消息",
            "预计",
            "同比",
            "小时",
            "美元",
        }
        tokens: set[str] = set()
        for raw in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", (text or "").lower()):
            if len(raw) >= 2 and not raw[0].isascii():
                # Split CJK runs into character bigrams for finer-grained matching
                for i in range(len(raw) - 1):
                    bigram = raw[i : i + 2]
                    if bigram not in stopwords:
                        tokens.add(bigram)
            else:
                if raw not in stopwords:
                    tokens.add(raw)
        return tokens

    def _title_overlap_score(self, summary: str, title: str) -> float:
        sum_tokens = self._match_tokens(summary)
        title_tokens = self._match_tokens(title)
        if not sum_tokens or not title_tokens:
            return 0.0
        return len(sum_tokens & title_tokens) / max(len(sum_tokens), 1)

    def _find_summarization_alignment_mismatch(
        self,
        *,
        article_id: int,
        one_line: str,
        title_by_id: Dict[int, str],
    ) -> Tuple[int, float, float] | None:
        own_title = title_by_id.get(article_id, "")
        own_score = self._title_overlap_score(one_line, own_title)
        if own_score >= 0.18:
            return None

        best_id = article_id
        best_score = own_score
        for candidate_id, candidate_title in title_by_id.items():
            score = self._title_overlap_score(one_line, candidate_title)
            if score > best_score:
                best_score = score
                best_id = candidate_id

        # Near-duplicate news clusters often share people, countries, and verbs.
        # Only fail when another title is clearly a much stronger match.
        if best_id != article_id and best_score >= 0.45 and best_score - own_score >= 0.25:
            return (best_id, own_score, best_score)
        return None

    def _char_units(self, ch: str) -> float:
        if not ch:
            return 0.0
        if ch.isspace():
            return 0.0

        eaw = unicodedata.east_asian_width(ch)
        if eaw in {"W", "F"}:
            return 1.0

        if ch.isascii():
            return 0.5

        return 1.0

    def _one_line_units(self, text: str) -> float:
        return sum(self._char_units(ch) for ch in (text or ""))

    def _trim_one_line_by_units(self, text: str, target_units: float) -> str:
        out: List[str] = []
        total = 0.0
        for ch in text or "":
            units = self._char_units(ch)
            if total + units > target_units:
                break
            out.append(ch)
            total += units

        trimmed = "".join(out).rstrip("，,、；;：:。.!?！？ ")
        if len(trimmed) < len(text or "") and trimmed:
            return f"{trimmed}…"
        return trimmed or (text or "")[:1]

    def _normalize_category(self, category: str) -> str:
        return (category or "").strip()

    def _collect_categories(
        self,
        articles: Sequence[ProcessedArticle],
    ) -> List[str]:
        present_order: List[str] = []
        present_set: set[str] = set()
        for item in articles:
            cat = (item.category or "").strip()
            if not cat:
                continue
            if cat in present_set:
                continue
            present_set.add(cat)
            present_order.append(cat)
        preferred = [
            category
            for category in getattr(self.cfg, "preferred_categories", [])
            if category in present_set
        ]
        return preferred + [category for category in present_order if category not in preferred]

    def _build_processed_articles(
        self,
        cleaned: Sequence[CleanedArticle],
        summary_map: Dict[int, Dict[str, str]],
    ) -> List[ProcessedArticle]:
        items: List[ProcessedArticle] = []
        for article in cleaned:
            mapped = summary_map.get(article.id)
            if not mapped:
                raise AIProcessingError(f"Summarization result missing id={article.id}")
            items.append(
                ProcessedArticle(
                    id=article.id,
                    title=article.title,
                    link=article.link,
                    pub_date=article.pub_date,
                    source=article.source,
                    one_line=mapped["one_line"],
                    category="",
                )
            )
        return items

    def _apply_category_map(
        self,
        articles: Sequence[ProcessedArticle],
        category_map: Dict[int, str],
    ) -> None:
        for article in articles:
            category = category_map.get(article.id)
            if category is None:
                raise AIProcessingError(f"Categorization result missing id={article.id}")
            article.category = category

    def _is_transient_error(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True

        name = exc.__class__.__name__.lower()
        msg = str(exc).lower()
        return any(
            keyword in name or keyword in msg
            for keyword in (
                "timeout",
                "connection",
                "temporar",
                "rate limit",
                "too many requests",
                "429",
                "bad gateway",
                "502",
            )
        )

    def _should_retry_call_error(self, exc: Exception) -> bool:
        if self._is_transient_error(exc):
            return True

        name = exc.__class__.__name__.lower()
        msg = str(exc).lower()
        for keyword in self.cfg.retry_error_keywords:
            normalized = str(keyword or "").strip().lower()
            if normalized and (normalized in name or normalized in msg):
                return True
        return False

    def _backoff_delay(self, attempt: int) -> float:
        idx = min(attempt, max(len(self.cfg.backoff_seconds) - 1, 0))
        base = float(self.cfg.backoff_seconds[idx]) if self.cfg.backoff_seconds else 1.0
        jitter = random.randint(0, max(self.cfg.jitter_ms_max, 0)) / 1000.0
        return base + jitter

    def _validate_overview(
        self,
        data: Any,
        valid_ids: set[int],
        locale: Locale,
    ) -> List[str]:
        lines = self._coerce_summary_lines(data)
        if not isinstance(lines, list):
            raise ValueError("summaryLines must be an array")

        cleaned = [str(x or "").strip() for x in lines if str(x or "").strip()]
        if not (4 <= len(cleaned) <= 9):
            raise ValueError(f"summaryLines must have 4-9 lines, got {len(cleaned)}")
        headline = self._validate_overview_headline(cleaned[0], locale=locale)
        groups = self._validate_overview_groups(
            cleaned[1:],
            valid_ids,
            locale=locale,
        )
        return [headline, *groups]

    def _validate_overview_headline(
        self,
        data: Any,
        *,
        locale: Locale,
    ) -> str:
        if isinstance(data, dict):
            headline = data.get("headline")
        else:
            headline = data

        text = str(headline or "").strip()
        if not text:
            raise ValueError("overview headline must not be empty")
        if text.startswith("{") or text.startswith("["):
            raise ValueError("overview headline must be plain text, not JSON")
        if "\n" in text:
            raise ValueError("overview headline must be one line")
        if re.search(r"\[(\d+)\]", text):
            raise ValueError("overview first line must not contain references")

        adjusted = self._apply_text_length_policy(
            text,
            label="summaryLine#1",
            value_fn=lambda x: float(len(x or "")),
            trim_fn=self._trim_summary_line_by_chars,
            hard_limit=self._summary_line_hard_limit,
            soft_limit=self._summary_line_soft_limit,
            trim_target=self._summary_line_target_len,
        )
        if len(adjusted) > int(self._summary_line_soft_limit):
            raise ValueError(
                f"overview line exceeds {int(self._summary_line_soft_limit)} chars and cannot be trimmed"
            )
        return adjusted

    def _validate_overview_groups(
        self,
        data: Any,
        valid_ids: set[int],
        *,
        locale: Locale,
    ) -> List[str]:
        lines = self._coerce_summary_lines(data)
        if not isinstance(lines, list):
            raise ValueError("summaryLines must be an array")

        cleaned = [str(x or "").strip() for x in lines if str(x or "").strip()]
        if not (3 <= len(cleaned) <= 8):
            raise ValueError(f"summaryLines must have 3-8 lines, got {len(cleaned)}")

        normalized: List[str] = []
        ref_re = re.compile(r"\[(\d+)\]")
        tail_ref_re = re.compile(r"(?:\[(\d+)\])+$")
        for idx, line in enumerate(cleaned, start=2):
            adjusted = self._apply_text_length_policy(
                line,
                label=f"summaryLine#{idx}",
                value_fn=lambda x: float(len(x or "")),
                trim_fn=self._trim_summary_line_by_chars,
                hard_limit=self._summary_line_hard_limit,
                soft_limit=self._summary_line_soft_limit,
                trim_target=self._summary_line_target_len,
            )
            if len(adjusted) > int(self._summary_line_soft_limit):
                raise ValueError(
                    f"overview line exceeds {int(self._summary_line_soft_limit)} chars and cannot be trimmed"
                )
            refs = ref_re.findall(adjusted)
            if not refs:
                raise ValueError("overview detail line must contain references")
            if not tail_ref_re.search(adjusted):
                raise ValueError("overview detail line must end with references")
            normalized.append(adjusted)
            for ref in refs:
                if int(ref) not in valid_ids:
                    raise ValueError(f"overview references non-existent article id: {ref}")
        return normalized

    def _postprocess_overview_lines(
        self,
        lines: List[str],
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> List[str]:
        out = [str(x or "").strip() for x in lines if str(x or "").strip()]
        if not out:
            raise ValueError("Overview is empty after postprocess")

        ref_re = re.compile(r"\[(\d+)\]")
        detail = out[1:]
        if detail:
            multi_ref_count = sum(1 for line in detail if len(ref_re.findall(line)) >= 2)
            min_multi = max(2, len(detail) // 2)
            if multi_ref_count < min_multi:
                raise ValueError("Overview style does not meet aggregation requirements (insufficient multi-ref lines)")

        if len(out) < 4:
            raise ValueError("Overview has too few lines (strict mode)")
        return out[:9]

    def _sanitize_overview_lines(
        self,
        lines: List[str],
        *,
        strip_first_line_refs: bool,
    ) -> List[str]:
        cleaned: List[str] = []
        ref_re = re.compile(r"\[(\d+)\]")
        artifact_re = re.compile(
            r"(?i)(websearch|web_search|search_with_snippets|tool_call|function_call|code_execution|\"query\"\\s*:)"
        )
        for idx, raw in enumerate(lines):
            line = re.sub(r"^\s*[\-\*\d\.\)\(、:：]+\s*", "", str(raw or "").strip())
            line = re.sub(
                r"</?(think|analysis|reasoning|thought)[^>]*>",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()
            line = line.replace("```", "").strip()
            if not line:
                continue

            lowered = line.lower()
            if lowered.startswith(("code_execution", "tool_call", "function_call")):
                continue
            if '"code"' in lowered and ("print(" in lowered or "len(" in lowered):
                continue
            if artifact_re.search(line):
                continue
            if line.startswith("[") and line.endswith("]"):
                continue

            refs = "".join([f"[{x}]" for x in ref_re.findall(line)])
            body = ref_re.sub("", line).strip()

            if strip_first_line_refs and idx == 0:
                refs = ""

            merged = f"{body}{refs}"
            merged = self._apply_text_length_policy(
                merged,
                label=f"summaryLine#{idx + 1}",
                value_fn=lambda x: float(len(x or "")),
                trim_fn=self._trim_summary_line_by_chars,
                hard_limit=self._summary_line_hard_limit,
                soft_limit=self._summary_line_soft_limit,
                trim_target=self._summary_line_target_len,
            )

            merged = merged.strip()
            if not merged:
                continue
            cleaned.append(merged)
        return cleaned

    def _normalize_overview_line_count(
        self,
        lines: List[str],
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> List[str]:
        out = [str(x or "").strip() for x in lines if str(x or "").strip()]
        if not out:
            return []
        if len(out) > 9:
            return out[:9]
        if len(out) >= 4:
            return out
        return out

    def _overview_fallback_text(
        self,
        key: str,
        default: str,
        locale: Locale,
    ) -> str:
        text = str(locale.fallback_texts.get(key, "") or "").strip()
        return text or default

    def _build_local_overview(
        self,
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> List[str]:
        if not articles:
            return [self._overview_fallback_text("no_overview", "No overview available today.", locale)]

        first_line = self._build_macro_overview_line(articles, locale=locale)
        detail_lines: List[str] = []
        idx = 0
        while idx < len(articles) and len(detail_lines) < 8:
            chunk = articles[idx : idx + 2]
            if not chunk:
                break
            line, _ = self._build_grouped_line(chunk, min_items=1, max_items=2)
            if line:
                detail_lines.append(line)
            idx += 2

        merged = [first_line] + detail_lines
        if len(merged) < 5:
            for item in articles:
                if len(merged) >= 5:
                    break
                merged.append(self._compose_single_line(item))
        return merged[:9]

    def _build_macro_overview_line(
        self,
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> str:
        is_english = locale.lang.lower().startswith("en")
        separator = ", " if is_english else "、"
        theme_counts = self._extract_theme_counts(articles, locale=locale)
        top_themes = [
            name for name, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True) if count > 0
        ][:2]
        article_count = len(articles)
        if top_themes:
            theme_text = separator.join(top_themes)
            if is_english:
                line = (
                    f"Today's key themes center on {theme_text}, with {article_count} notable developments "
                    "interacting across markets and sentiment shifting quickly."
                )
            else:
                line = f"今日主线集中在{theme_text}，共{article_count}条重点动态交织推进，市场情绪与风险偏好快速切换。"
        else:
            if is_english:
                line = (
                    f"Today's news flow spans {article_count} notable developments, with cross-market signals "
                    "interacting and volatility moving alongside opportunity."
                )
            else:
                line = f"今日共有{article_count}条重点动态，跨市场消息交叉影响，结构性机会与波动并存。"
        limit = int(self._summary_line_target_len)
        if len(line) > limit:
            line = self._condense_fragment(line, limit)
        suffix = "." if is_english else "。"
        return line.rstrip("，,、；;：:。.!?！？ ") + suffix

    def _extract_theme_counts(
        self,
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> Dict[str, int]:
        text = " ".join(
            [
                self._normalize_summary_fragment(item.one_line or item.title)
                for item in articles
                if (item.one_line or item.title)
            ]
        )
        theme_keywords: Dict[str, List[str]] = locale.theme_keywords
        lower = text.lower()
        counts: Dict[str, int] = {key: 0 for key in theme_keywords}
        for theme, keywords in theme_keywords.items():
            for keyword in keywords:
                counts[theme] += lower.count(keyword.lower())
        return counts

    def _normalize_summary_fragment(self, text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        s = re.sub(r"\[(\d+)\]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"^[\-\*\d\.\)\s]+", "", s)
        return s.rstrip("，,、；;：:。.!?！？ ")

    def _build_grouped_line(
        self,
        items: List[ProcessedArticle],
        min_items: int = 2,
        max_items: int = 3,
    ) -> Tuple[str, List[int]]:
        chosen_parts: List[str] = []
        chosen_ids: List[int] = []
        limit = int(self._summary_line_target_len)

        for item in items[:max_items]:
            fragment = self._normalize_summary_fragment(item.one_line or item.title)
            if not fragment:
                continue

            trial_parts = chosen_parts + [fragment]
            trial_ids = chosen_ids + [item.id]
            refs = "".join([f"[{x}]" for x in trial_ids])
            trial_line = f"{'；'.join(trial_parts)}{refs}"
            if len(trial_line) > limit and chosen_parts:
                break
            if len(trial_line) > limit:
                max_body_len = limit - len(refs)
                compact = self._condense_fragment(fragment, max_body_len)
                if compact:
                    chosen_parts = [compact]
                    chosen_ids = [item.id]
                break
            chosen_parts = trial_parts
            chosen_ids = trial_ids

        if len(chosen_ids) < min_items:
            return "", []

        refs = "".join([f"[{x}]" for x in chosen_ids])
        line = f"{'；'.join(chosen_parts)}{refs}"
        if len(line) > limit:
            line = self._trim_line_keep_refs("；".join(chosen_parts), refs, limit)
        return line, chosen_ids

    def _compose_single_line(self, item: ProcessedArticle) -> str:
        fragment = self._normalize_summary_fragment(item.one_line or item.title)
        line = f"{fragment}[{item.id}]"
        limit = int(self._summary_line_target_len)
        if len(line) > limit:
            max_body_len = limit - len(f"[{item.id}]")
            compact = self._condense_fragment(fragment, max_body_len)
            line = f"{compact}[{item.id}]"
        return line

    def _trim_line_keep_refs(self, body: str, refs: str, max_len: int) -> str:
        max_body_len = max(max_len - len(refs), 0)
        compact = self._condense_fragment(body or "", max_body_len)
        return f"{compact}{refs}"

    def _trim_summary_line_by_chars(self, text: str, target_len: float) -> str:
        max_len = max(int(target_len), 1)
        ref_re = re.compile(r"\[(\d+)\]")
        refs = "".join([f"[{x}]" for x in ref_re.findall(text or "")])
        body = ref_re.sub("", text or "").strip()
        if refs:
            return self._trim_line_keep_refs(body, refs, max_len)
        return self._condense_fragment(body, max_len)

    def _condense_fragment(self, text: str, max_len: int) -> str:
        s = (text or "").strip()
        if max_len <= 0:
            return ""
        if len(s) <= max_len:
            return s

        parts = [p.strip() for p in re.split(r"[，；。,.!?！？:：]", s) if p.strip()]
        for part in parts:
            if len(part) <= max_len:
                return part

        if max_len <= 4:
            return s[:max_len]
        return s[: max_len - 1].rstrip("，,、；;：:。.!?！？ ") + "…"

    def _coerce_summary_lines(self, data: Any, depth: int = 0) -> List[str] | None:
        if depth > 4:
            return None

        if isinstance(data, list):
            return [str(x or "").strip() for x in data if str(x or "").strip()]

        if isinstance(data, dict):
            preferred_keys = (
                "summaryLines",
                "groupLines",
                "overviewLines",
                "lines",
                "summary",
                "result",
                "data",
                "output",
                "text",
                "content",
                "message",
                "response",
            )
            for key in preferred_keys:
                if key in data:
                    lines = self._coerce_summary_lines(data.get(key), depth + 1)
                    if lines:
                        return lines

            content = self._transport.extract_chat_content_from_dict(data)
            if content:
                lines = self._coerce_summary_lines(content, depth + 1)
                if lines:
                    return lines
            return None

        if isinstance(data, str):
            text = data.strip()
            if not text:
                return None

            parsed = self._transport.try_parse_json_payload(text)
            if parsed is not None and parsed is not data:
                lines = self._coerce_summary_lines(parsed, depth + 1)
                if lines:
                    return lines
            return [x.strip() for x in text.splitlines() if x.strip()]

        return None

    async def _request_summarization(
        self,
        articles: List[CleanedArticle],
        locale: Locale,
        target: AIRetryTarget,
    ) -> Dict[str, Any]:
        user_payload = [
            {
                "id": x.id,
                "title": x.title,
                "content": x.content,
            }
            for x in articles
        ]
        articles_json = json_dumps(user_payload)
        example_output = '{\n  "perArticle":[{"id":1,"oneLine":"..."}]\n}'
        is_english = locale.lang.lower().startswith("en")
        system_prompt = self._render_prompt_with_default(
            step="summarization",
            role="system",
            default=(
                "Task: output oneLine for each news item.\n"
                "oneLine: a single concise English line with only the core information.\n"
                "Each oneLine must match its own item only. Do not reuse wording from another article or merge similar items.\n"
                "Output strict JSON only. No explanation, no Markdown."
                if is_english
                else "任务：为每篇新闻输出 oneLine。\n"
                "oneLine：中文单行，简洁明确，只保留核心信息。\n"
                "每条 oneLine 只能对应同 id 的原文，不要复用别条新闻的表述，不要把相似新闻合并成一句。\n"
                "只输出严格 JSON，不要解释、不要 Markdown。"
            ),
        )
        user_prompt = self._render_prompt_with_default(
            step="summarization",
            role="user",
            default=(
                "Please process the following {count} news items.\n"
                "Output format:\n"
                "{example_output}\n\n"
                "articles={articles_json}"
                if is_english
                else "请处理以下 {count} 篇新闻。\n"
                "输出格式：\n"
                "{example_output}\n\n"
                "articles={articles_json}"
            ),
            count=len(articles),
            example_output=example_output,
            articles_json=articles_json,
        )

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "perArticle": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer"},
                            "oneLine": {"type": "string"},
                        },
                        "required": ["id", "oneLine"],
                    },
                },
            },
            "required": ["perArticle"],
        }

        raw = await self._transport.request_json(
            system_prompt,
            user_prompt,
            schema,
            "summarization",
            target=target,
        )
        if not isinstance(raw, dict):
            raise ValueError("Summarization output is not a JSON object")
        return raw

    async def _request_categorization(
        self,
        articles: List[CleanedArticle],
        locale: Locale,
        target: AIRetryTarget,
        *,
        category_candidates: Sequence[str],
        used_categories: Sequence[str],
        total_article_count: int,
    ) -> Dict[str, Any]:
        preferred_categories = self._configured_preferred_categories()
        suggested_categories = []
        if not preferred_categories:
            suggested_categories = [str(item).strip() for item in category_candidates if str(item).strip()]
        user_payload = [
            {
                "id": x.id,
                "title": x.title,
                "content": self._categorization_content(x.content),
            }
            for x in articles
        ]
        articles_json = json_dumps(user_payload)
        example_output = '{\n  "perArticle":[{"id":1,"category":"..."}]\n}'
        prompt_kwargs = self._build_categorization_prompt_kwargs(
            locale=locale,
            shard_article_count=len(articles),
            total_article_count=total_article_count,
            preferred_categories=preferred_categories,
            suggested_categories=suggested_categories,
            used_categories=used_categories,
            example_output=example_output,
            articles_json=articles_json,
        )
        is_english = locale.lang.lower().startswith("en")
        system_prompt = self._render_prompt_with_default(
            step="categorization",
            role="system",
            default=(
                "Task: output exactly one category for each news item.\n"
                "Use the title and short content excerpt to judge the category.\n"
                "{category_language_instruction}\n"
                "{categorization_policy_instruction}\n"
                "{already_used_categories_instruction}"
                "The category must not be empty.\n"
                "Output strict JSON only. No explanation, no Markdown."
                if is_english
                else "任务：为每篇新闻输出 1 个 category。\n"
                "可参考标题和正文片段判断。\n"
                "{category_language_instruction}\n"
                "{categorization_policy_instruction}\n"
                "{already_used_categories_instruction}"
                "category 不能为空。\n"
                "只输出严格 JSON，不要解释、不要 Markdown。"
            ),
            **prompt_kwargs,
        )
        user_prompt = self._render_prompt_with_default(
            step="categorization",
            role="user",
            default=(
                "Please process the following {count} news items.\n"
                "{category_candidates_block}"
                "{already_used_categories_block}"
                "Output format:\n"
                "{example_output}\n\n"
                "articles={articles_json}"
                if is_english
                else "请处理以下 {count} 篇新闻。\n"
                "{category_candidates_block}"
                "{already_used_categories_block}"
                "输出格式：\n"
                "{example_output}\n\n"
                "articles={articles_json}"
            ),
            **prompt_kwargs,
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "perArticle": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer"},
                            "category": {"type": "string"},
                        },
                        "required": ["id", "category"],
                    },
                },
            },
            "required": ["perArticle"],
        }
        raw = await self._transport.request_json(
            system_prompt,
            user_prompt,
            schema,
            "categorization",
            target=target,
        )
        if not isinstance(raw, dict):
            raise ValueError("Categorization output is not a JSON object")
        return raw

    async def _request_category_suggestion(
        self,
        articles: List[CleanedArticle],
        locale: Locale,
        target: AIRetryTarget,
        *,
        existing_categories: Sequence[str],
        total_article_count: int,
    ) -> Dict[str, Any]:
        user_payload = [
            {
                "title": x.title,
                "content": self._category_suggestion_content(x.content),
            }
            for x in articles
        ]
        articles_json = json_dumps(user_payload)
        example_output = '{\n  "categories":["国际","财经","科技"]\n}'
        prompt_kwargs = self._build_category_suggestion_prompt_kwargs(
            locale=locale,
            shard_article_count=len(articles),
            total_article_count=total_article_count,
            existing_categories=existing_categories,
            example_output=example_output,
            articles_json=articles_json,
        )
        is_english = locale.lang.lower().startswith("en")
        system_prompt = self._render_prompt_with_default(
            step="category_suggestion",
            role="system",
            default=(
                "Task: propose a compact set of categories for this batch of news.\n"
                "Use the title and short content excerpt to judge the categories.\n"
                "{category_language_instruction}\n"
                "{category_suggestion_policy_instruction}\n"
                "{existing_categories_instruction}"
                "The category list must not be empty.\n"
                "Output strict JSON only. No explanation, no Markdown."
                if is_english
                else "任务：先为这一批新闻提炼一组建议分类。\n"
                "可参考标题和正文片段判断。\n"
                "{category_language_instruction}\n"
                "{category_suggestion_policy_instruction}\n"
                "{existing_categories_instruction}"
                "分类列表不能为空。\n"
                "只输出严格 JSON，不要解释、不要 Markdown。"
            ),
            **prompt_kwargs,
        )
        user_prompt = self._render_prompt_with_default(
            step="category_suggestion",
            role="user",
            default=(
                "Please process the following {count} news items.\n"
                "{existing_categories_block}"
                "Output format:\n"
                "{example_output}\n\n"
                "articles={articles_json}"
                if is_english
                else "请处理以下 {count} 篇新闻。\n"
                "{existing_categories_block}"
                "输出格式：\n"
                "{example_output}\n\n"
                "articles={articles_json}"
            ),
            **prompt_kwargs,
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["categories"],
        }
        raw = await self._transport.request_json(
            system_prompt,
            user_prompt,
            schema,
            "category_suggestion",
            target=target,
        )
        if not isinstance(raw, dict):
            raise ValueError("Category suggestion output is not a JSON object")
        return raw

    async def _request_overview_headline(
        self,
        articles: List[ProcessedArticle],
        locale: Locale,
        target: AIRetryTarget,
    ) -> Dict[str, Any]:
        summary_target = int(self._summary_line_target_len)
        user_payload = [
            {
                "title": x.title,
                "oneLine": x.one_line,
            }
            for x in articles
        ]
        article_lines = json_dumps(user_payload)
        example_output = locale.get("prompts.overview_headline.example_output", '{"headline":"..."}')
        system_prompt = self._render_prompt_with_default(
            step="overview_headline",
            role="structured",
            default=(
                "Task: write line 1 of the news overview from articles. Write in English even if articles are in other languages.\n"
                'Return JSON only: {"headline":"..."}\n'
                "Requirements:\n"
                "- Write exactly one line.\n"
                "- It must be the daily overview and mention at least one eye-catching specific event.\n"
                "- [id] references are forbidden.\n"
                "- Keep it natural, like the opening sentence of a real digest, not a title.\n"
                "- Keep it around {summary_target} characters when possible.\n"
                "- Output nothing except the JSON object."
                if locale.lang.lower().startswith("en")
                else "任务：根据 articles 写新闻总览的第1行。用中文撰写。\n"
                '只返回 JSON：{"headline":"..."}\n'
                "要求：\n"
                "- 只写 1 行。\n"
                "- 这行是全天总述，要点出至少一个最吸引眼球的具体事件。\n"
                "- 禁止出现任何 `[id]` 引用。\n"
                "- 语气自然，像真正的日报开头，不要写成标题。\n"
                "- 尽量控制在 {summary_target} 字左右。\n"
                "- 除 JSON 外不要输出任何内容。"
            ),
            summary_target=summary_target,
        )
        user_prompt = self._render_prompt_with_default(
            step="overview_headline",
            role="user",
            default=(
                "Article count: {count}\n"
                "Please generate line 1 only.\n"
                "Output format:\n"
                "{example_output}\n\n"
                "articles={article_lines}"
                if locale.lang.lower().startswith("en")
                else "文章数: {count}\n"
                "请只生成第1行。\n"
                "输出格式：\n"
                "{example_output}\n\n"
                "articles={article_lines}"
            ),
            count=len(articles),
            article_lines=article_lines,
            example_output=example_output,
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "headline": {"type": "string"},
            },
            "required": ["headline"],
        }
        raw = await self._transport.request_json(
            system_prompt,
            user_prompt,
            schema,
            "overview_headline",
            target=target,
        )
        if not isinstance(raw, dict):
            raise ValueError("Overview headline output is not a JSON object")
        return raw

    async def _request_overview_groups(
        self,
        articles: List[ProcessedArticle],
        *,
        headline: str,
        locale: Locale,
        target: AIRetryTarget,
    ) -> Dict[str, Any]:
        summary_target = int(self._summary_line_target_len)
        user_payload = [
            {
                "id": x.id,
                "title": x.title,
                "oneLine": x.one_line,
            }
            for x in articles
        ]
        article_lines = json_dumps(user_payload)
        example_output = locale.get("prompts.overview_groups.example_output", '{"groupLines":["... [1][2]"]}')
        max_id = max((item.id for item in articles), default=0)
        system_prompt = self._render_prompt_with_default(
            step="overview_groups",
            role="structured",
            default=(
                "Task: write lines 2 onward of the news overview from articles. Write in English even if articles are in other languages.\n"
                'Return JSON only: {"groupLines":[...]}\n'
                "Requirements:\n"
                "- Output 3-8 lines.\n"
                "- Each line must be exactly one sentence and must not contain newline characters.\n"
                "- Continue smoothly from the given headline, but do not repeat what the headline already says.\n"
                "- Each line should synthesize 2-4 news items, not a list of themes or keywords.\n"
                "- Every line must end with [id] references; multiple references are allowed.\n"
                "- Prefer using 2 or more [id] references on most lines.\n"
                "- Keep each line around {summary_target} characters when possible.\n"
                "- Output nothing except the JSON object."
                if locale.lang.lower().startswith("en")
                else "任务：根据 articles 写新闻总览的后续几行。用中文撰写。\n"
                '只返回 JSON：{"groupLines":[...]}\n'
                "要求：\n"
                "- 输出 3-8 行。\n"
                "- 每行单独成句，字符串内部不得出现换行符。\n"
                "- 从第2行开始续写，与给定 headline 保持顺滑衔接，但不要重复 headline 已经说过的话。\n"
                "- 每行归纳 2-4 条新闻，不要只列主题词或关键词。\n"
                "- 每行行尾都要附 `[id]` 引用，可多个。\n"
                "- 尽量让多数行同时包含 2 个及以上 `[id]`。\n"
                "- 每行尽量控制在 {summary_target} 字左右。\n"
                "- 除 JSON 外不要输出任何内容。"
            ),
            summary_target=summary_target,
        )
        user_prompt = self._render_prompt_with_default(
            step="overview_groups",
            role="user",
            default=(
                "Article count: {count}\n"
                "Valid reference id range: 1-{max_id}\n"
                "headline={headline}\n"
                "Please generate lines 2 onward.\n"
                "Output format:\n"
                "{example_output}\n\n"
                "articles={article_lines}"
                if locale.lang.lower().startswith("en")
                else "文章数: {count}\n"
                "可引用 id 范围: 1-{max_id}\n"
                "headline={headline}\n"
                "请生成第2行起的总览。\n"
                "输出格式：\n"
                "{example_output}\n\n"
                "articles={article_lines}"
            ),
            count=len(articles),
            max_id=max_id,
            headline=headline,
            article_lines=article_lines,
            example_output=example_output,
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "groupLines": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["groupLines"],
        }
        raw = await self._transport.request_json(
            system_prompt,
            user_prompt,
            schema,
            "overview_groups",
            target=target,
        )
        if not isinstance(raw, dict):
            raise ValueError("Overview groups output is not a JSON object")
        return raw

    async def _request_overview_headline_text(
        self,
        articles: List[ProcessedArticle],
        locale: Locale,
        target: AIRetryTarget,
    ) -> str:
        summary_target = int(self._summary_line_target_len)
        user_payload = [
            {
                "title": x.title,
                "oneLine": x.one_line,
            }
            for x in articles
        ]
        article_lines = json_dumps(user_payload)
        example_output = locale.get(
            "prompts.overview_headline.plain_text_example_output",
            (
                "<one-line overview>"
                if locale.lang.lower().startswith("en")
                else "<一句总述>"
            ),
        )
        system_prompt = self._render_prompt_with_default(
            step="overview_headline",
            role="plain_text",
            default=(
                "Task: write line 1 of the news overview from articles as plain text. Write in English even if articles are in other languages.\n"
                "Requirements:\n"
                "- Write exactly one line.\n"
                "- It must be the daily overview and mention at least one eye-catching specific event.\n"
                "- [id] references are forbidden.\n"
                "- Keep it natural, like the opening sentence of a real digest, not a title.\n"
                "- Keep it around {summary_target} characters when possible.\n"
                "- Do not output JSON.\n"
                "- No title, no explanation, no prefix or suffix."
                if locale.lang.lower().startswith("en")
                else "任务：根据 articles 输出新闻总览的第1行纯文本。用中文撰写。\n"
                "要求：\n"
                "- 只写 1 行。\n"
                "- 这行是全天总述，要点出至少一个最吸引眼球的具体事件。\n"
                "- 禁止出现任何 `[id]` 引用。\n"
                "- 语气自然，像真正的日报开头，不要写成标题。\n"
                "- 尽量控制在 {summary_target} 字左右。\n"
                "- 不要输出 JSON。\n"
                "- 不要输出标题、前后缀或解释。"
            ),
            summary_target=summary_target,
        )
        user_prompt = self._render_prompt_with_default(
            step="overview_headline",
            role="user",
            default=(
                "Article count: {count}\n"
                "Please generate line 1 only.\n"
                "Output format:\n"
                "{example_output}\n\n"
                "articles={article_lines}"
                if locale.lang.lower().startswith("en")
                else "文章数: {count}\n"
                "请只生成第1行。\n"
                "输出格式：\n"
                "{example_output}\n\n"
                "articles={article_lines}"
            ),
            count=len(articles),
            article_lines=article_lines,
            example_output=example_output,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._transport.chat(messages, response_format=None, target=target)

    async def _request_overview_groups_text(
        self,
        articles: List[ProcessedArticle],
        *,
        headline: str,
        locale: Locale,
        target: AIRetryTarget,
    ) -> str:
        summary_target = int(self._summary_line_target_len)
        user_payload = [
            {
                "id": x.id,
                "title": x.title,
                "oneLine": x.one_line,
            }
            for x in articles
        ]
        article_lines = json_dumps(user_payload)
        max_id = max((item.id for item in articles), default=0)
        example_output = locale.get(
            "prompts.overview_groups.example_output",
            '{"groupLines":["... [1][2]"]}',
        )
        system_prompt = self._render_prompt_with_default(
            step="overview_groups",
            role="plain_text",
            default=(
                "Task: write lines 2 onward of the news overview as plain text from articles. Write in English even if articles are in other languages.\n"
                "Requirements:\n"
                "- Output 3-8 lines.\n"
                "- Each line must be exactly one sentence and must not contain newline characters.\n"
                "- Continue smoothly from the given headline, but do not repeat what the headline already says.\n"
                "- Each line should synthesize 2-4 news items, not a list of themes or keywords.\n"
                "- Every line must end with [id] references; multiple references are allowed.\n"
                "- Prefer using 2 or more [id] references on most lines.\n"
                "- Keep each line around {summary_target} characters when possible.\n"
                "- No title, no explanation, no prefix or suffix."
                if locale.lang.lower().startswith("en")
                else "任务：根据 articles 输出新闻总览第2行起的纯文本。用中文撰写。\n"
                "要求：\n"
                "- 输出 3-8 行。\n"
                "- 每行单独成句，字符串内部不得出现换行符。\n"
                "- 从第2行开始续写，与给定 headline 保持顺滑衔接，但不要重复 headline 已经说过的话。\n"
                "- 每行归纳 2-4 条新闻，不要只列主题词或关键词。\n"
                "- 每行行尾都要附 `[id]` 引用，可多个。\n"
                "- 尽量让多数行同时包含 2 个及以上 `[id]`。\n"
                "- 每行尽量控制在 {summary_target} 字左右。\n"
                "- 不要输出标题、前后缀或解释。"
            ),
            summary_target=summary_target,
        )
        user_prompt = self._render_prompt_with_default(
            step="overview_groups",
            role="user",
            default=(
                "Article count: {count}\n"
                "Valid reference id range: 1-{max_id}\n"
                "headline={headline}\n"
                "Please generate lines 2 onward.\n"
                "Output format:\n"
                "{example_output}\n\n"
                "articles={article_lines}"
                if locale.lang.lower().startswith("en")
                else "文章数: {count}\n"
                "可引用 id 范围: 1-{max_id}\n"
                "headline={headline}\n"
                "请生成第2行起的总览。\n"
                "输出格式：\n"
                "{example_output}\n\n"
                "articles={article_lines}"
            ),
            count=len(articles),
            max_id=max_id,
            headline=headline,
            article_lines=article_lines,
            example_output=example_output,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._transport.chat(messages, response_format=None, target=target)


__all__ = ["AIProcessingError", "AIProcessor"]
