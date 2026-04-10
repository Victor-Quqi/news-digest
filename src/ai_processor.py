from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Awaitable, Callable, Sequence, TypeVar

import tiktoken

from .ai_debug import AIDebugSink
from .ai_outputs import AIOutputProcessor
from .ai_processor_types import AIProcessingError
from .ai_prompts import AIPromptBuilder, ChatRequestSpec, JSONRequestSpec
from .ai_transport import AITransport
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
    ) -> None:
        self.cfg = ai_config
        self.logger = logger
        self.locale = locale
        self._timer = timer or PipelineTimer(enabled=False)
        self._debug = AIDebugSink(ai_config, logger, capture_all=debug_capture_all)
        self._transport = AITransport(ai_config, env_config, logger, self._debug)

        one_line_hard_units = max(float(self.cfg.one_line_hard_units), 1.0)
        one_line_soft_units = max(float(self.cfg.one_line_soft_units), one_line_hard_units)
        one_line_trim_target_units = min(
            max(float(self.cfg.one_line_trim_target_units), 1.0),
            one_line_soft_units,
        )
        summary_line_target_len = max(int(self.cfg.summary_line_target_len), 1)
        summary_line_hard_limit = max(float(self.cfg.summary_line_hard_len), float(summary_line_target_len))
        summary_line_soft_limit = max(float(self.cfg.summary_line_soft_len), summary_line_hard_limit)

        self._prompt_builder = AIPromptBuilder(
            locale=locale,
            preferred_categories=self.cfg.preferred_categories,
            summary_line_target_len=summary_line_target_len,
        )
        self._outputs = AIOutputProcessor(
            logger=logger,
            locale=locale,
            preferred_categories=self.cfg.preferred_categories,
            one_line_hard_units=one_line_hard_units,
            one_line_soft_units=one_line_soft_units,
            one_line_trim_target_units=one_line_trim_target_units,
            summary_line_target_len=summary_line_target_len,
            summary_line_hard_limit=summary_line_hard_limit,
            summary_line_soft_limit=summary_line_soft_limit,
        )

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
        self.logger.debug("AI HTTP proxy from env: %s", env_config.openai_use_env_proxy)

    async def process_articles(self, articles: list[CleanedArticle]) -> ProcessedResult:
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
        categorization_task = asyncio.create_task(self._run_categorization(articles, threshold=threshold))

        try:
            summary_map = await summary_task
        except Exception:
            await self._cancel_task(categorization_task)
            raise

        with self._timer.stage("  Build articles"):
            processed_articles = self._outputs.build_processed_articles(articles, summary_map)

        try:
            summary_lines = await self._run_overview(processed_articles)
        except Exception:
            await self._cancel_task(categorization_task)
            raise

        categories: list[str] = []
        try:
            category_map = await categorization_task
        except AIProcessingError as exc:
            if self._categorization_is_strict():
                raise AIProcessingError(f"Categorization failed (strict mode): {exc}") from exc
            self.logger.warning("Categorization failed, continuing without category grouping: %s", exc)
            category_map = None

        if category_map:
            with self._timer.stage("  Categories"):
                self._outputs.apply_category_map(processed_articles, category_map)
                categories = self._outputs.collect_categories(processed_articles)

        return ProcessedResult(
            articles=processed_articles,
            categories=categories,
            summary_lines=summary_lines,
            degraded=False,
            warnings=[],
        )

    def build_degraded_result(
        self,
        articles: list[CleanedArticle],
        warning_text: str,
    ) -> ProcessedResult:
        resolved_warning_text = str(warning_text or "").strip() or self._locale_fallback_text("warning", "")
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
            summary_lines=[
                self._locale_fallback_text(
                    "overview_failed",
                    "AI overview generation failed. Original article list shown below.",
                )
            ],
            degraded=True,
            warnings=[resolved_warning_text] if resolved_warning_text else [],
        )

    def _locale_fallback_text(self, key: str, default: str) -> str:
        text = str(self.locale.fallback_texts.get(key, "") or "").strip()
        return text or default

    async def aclose(self) -> None:
        await self._transport.aclose()

    def _dump_debug(self, event: str, payload: dict[str, Any], force: bool = False) -> None:
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

    def _targets_for_step(self, step: str) -> list[AIRetryTarget]:
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
    ) -> tuple[int, int, bool]:
        threshold = max(int(self.cfg.retry_target_failure_threshold), 1)
        updated_failures = candidate_failures + 1
        if updated_failures >= threshold and candidate_index < len(targets) - 1:
            return candidate_index + 1, 0, True
        return candidate_index, updated_failures, False

    def _target_payload(self, target: AIRetryTarget) -> dict[str, str]:
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

    def _estimate_tokens(self, articles: Sequence[CleanedArticle]) -> int:
        payload = [
            {
                "id": article.id,
                "title": article.title,
                "content": article.content,
            }
            for article in articles
        ]
        return self._estimate_payload_tokens(payload)

    def _estimate_tokens_for_categorization(self, articles: Sequence[CleanedArticle]) -> int:
        payload = [
            {
                "id": article.id,
                "title": article.title,
                "content": self._prompt_builder.categorization_content(article.content),
            }
            for article in articles
        ]
        return self._estimate_payload_tokens(payload)

    def _estimate_tokens_for_category_suggestion(self, articles: Sequence[CleanedArticle]) -> int:
        payload = [
            {
                "title": article.title,
                "content": self._prompt_builder.category_suggestion_content(article.content),
            }
            for article in articles
        ]
        return self._estimate_payload_tokens(payload)

    def _estimate_payload_tokens(self, payload: Any) -> int:
        text = json_dumps(payload)
        if self.encoding is None:
            estimated = int(len(text) * 1.2)
            return estimated + 2000 + self.cfg.max_tokens
        return len(self.encoding.encode(text)) + 2000 + self.cfg.max_tokens

    def _build_shards(self, articles: Sequence[CleanedArticle]) -> list[list[CleanedArticle]]:
        shards: list[list[CleanedArticle]] = []
        current: list[CleanedArticle] = []
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

    def _build_categorization_shards(
        self,
        articles: Sequence[CleanedArticle],
    ) -> list[list[CleanedArticle]]:
        shards: list[list[CleanedArticle]] = []
        current: list[CleanedArticle] = []
        current_chars = 0

        for article in articles:
            article_chars = len(article.title) + len(self._prompt_builder.categorization_content(article.content))
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
    ) -> list[list[CleanedArticle]]:
        shards: list[list[CleanedArticle]] = []
        current: list[CleanedArticle] = []
        current_chars = 0

        for article in articles:
            article_chars = len(article.title) + len(self._prompt_builder.category_suggestion_content(article.content))
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
    ) -> dict[int, dict[str, str]]:
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
    ) -> dict[int, str]:
        category_candidates = self._prompt_builder.configured_preferred_categories()
        if not category_candidates:
            try:
                category_candidates = await self._run_category_suggestion(articles, threshold=threshold)
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

        merged: dict[int, str] = {}
        used_categories: list[str] = []
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

    async def _run_category_suggestion(
        self,
        articles: Sequence[CleanedArticle],
        *,
        threshold: int,
    ) -> list[str]:
        estimated_tokens = self._estimate_tokens_for_category_suggestion(articles)
        max_categories = self._prompt_builder.categorization_max_categories(len(articles))
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

        merged: list[str] = []
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

    async def _run_overview(self, articles: list[ProcessedArticle]) -> list[str]:
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
                            return self._outputs.build_local_overview(articles)
                    raise AIProcessingError(
                        f"Overview text fallback failed (strict mode, local fallback disabled): {fallback_exc}"
                    ) from fallback_exc
            if self.cfg.overview_local_fallback:
                self.logger.warning("Overview generation failed, using local rule-based overview: %s", exc)
                with self._timer.stage("  Overview local fallback"):
                    return self._outputs.build_local_overview(articles)
            raise AIProcessingError(f"Overview generation failed (strict mode): {exc}") from exc

    def _categorization_is_strict(self) -> bool:
        return bool(getattr(self.cfg, "categorization_strict", True))

    async def _summarize_shards(
        self,
        shards: Sequence[Sequence[CleanedArticle]],
    ) -> dict[int, dict[str, str]]:
        async def run_single(
            shard: Sequence[CleanedArticle],
            label: str,
        ) -> dict[int, dict[str, str]]:
            async with self._semaphore:
                async with self._timer.async_stage(f"    {label}"):
                    return await self._summarize_batch(list(shard), shard_label=label)

        results = await asyncio.gather(*(run_single(shard, f"Shard {index + 1}") for index, shard in enumerate(shards)))
        merged: dict[int, dict[str, str]] = {}
        for item in results:
            merged.update(item)
        return merged

    async def _summarize_batch(
        self,
        articles: list[CleanedArticle],
        *,
        shard_label: str = "",
    ) -> dict[int, dict[str, str]]:
        expected_ids = {article.id for article in articles}
        title_by_id = {article.id: article.title for article in articles}
        prefix = f"      {shard_label} " if shard_label else "    "
        request = self._prompt_builder.build_summarization_request(articles)
        return await self._run_json_phase(
            step=request.step,
            log_label="Summarization",
            request_fn=lambda target: self._request_json(request, target=target),
            validate_fn=lambda data: self._outputs.normalize_summarization_result(
                data,
                expected_ids=expected_ids,
                title_by_id=title_by_id,
            ),
            timer_api_label=f"{prefix}API",
            timer_validate_label=f"{prefix}Validate",
        )

    async def _categorize_batch(
        self,
        articles: list[CleanedArticle],
        *,
        shard_label: str = "",
        category_candidates: Sequence[str],
        used_categories: Sequence[str],
        total_article_count: int,
    ) -> dict[int, str]:
        expected_ids = {article.id for article in articles}
        prefix = f"      {shard_label} " if shard_label else "    "
        request = self._prompt_builder.build_categorization_request(
            articles,
            category_candidates=category_candidates,
            used_categories=used_categories,
            total_article_count=total_article_count,
        )
        return await self._run_json_phase(
            step=request.step,
            log_label="Categorization",
            request_fn=lambda target: self._request_json(request, target=target),
            validate_fn=lambda data: self._outputs.normalize_categorization_result(
                data,
                expected_ids=expected_ids,
            ),
            timer_api_label=f"{prefix}Category API",
            timer_validate_label=f"{prefix}Category Validate",
        )

    async def _suggest_categories_batch(
        self,
        articles: list[CleanedArticle],
        *,
        shard_label: str = "",
        existing_categories: Sequence[str],
        total_article_count: int,
    ) -> list[str]:
        prefix = f"      {shard_label} " if shard_label else "    "
        request = self._prompt_builder.build_category_suggestion_request(
            articles,
            existing_categories=existing_categories,
            total_article_count=total_article_count,
        )
        max_categories = self._prompt_builder.categorization_max_categories(total_article_count)
        return await self._run_json_phase(
            step=request.step,
            log_label="Category suggestion",
            request_fn=lambda target: self._request_json(request, target=target),
            validate_fn=lambda data: self._outputs.normalize_category_suggestion_result(
                data,
                max_categories=max_categories,
            ),
            timer_api_label=f"{prefix}Suggest API",
            timer_validate_label=f"{prefix}Suggest Validate",
        )

    async def _generate_overview(self, articles: list[ProcessedArticle]) -> list[str]:
        headline = await self._generate_overview_headline(articles)
        groups = await self._generate_overview_groups(articles, headline=headline)
        return self._outputs.postprocess_overview_lines([headline, *groups], articles)

    async def _generate_overview_headline(
        self,
        articles: list[ProcessedArticle],
    ) -> str:
        request = self._prompt_builder.build_overview_headline_request(articles)
        return await self._run_json_phase(
            step=request.step,
            log_label="Overview headline",
            request_fn=lambda target: self._request_json(request, target=target),
            validate_fn=lambda data: self._outputs.validate_overview_headline(data),
            timer_api_label="    Headline API",
            timer_validate_label="    Headline Validate",
        )

    async def _generate_overview_groups(
        self,
        articles: list[ProcessedArticle],
        *,
        headline: str,
    ) -> list[str]:
        valid_ids = {article.id for article in articles}
        request = self._prompt_builder.build_overview_groups_request(articles, headline=headline)
        return await self._run_json_phase(
            step=request.step,
            log_label="Overview groups",
            request_fn=lambda target: self._request_json(request, target=target),
            validate_fn=lambda data: self._outputs.validate_overview_groups(data, valid_ids),
            timer_api_label="    Groups API",
            timer_validate_label="    Groups Validate",
        )

    async def _generate_overview_text_fallback(
        self,
        articles: list[ProcessedArticle],
    ) -> list[str]:
        valid_ids = {article.id for article in articles}
        headline_request = self._prompt_builder.build_overview_headline_text_request(articles)
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
                headline_text = await self._request_chat(headline_request, target=target)
                headline = self._outputs.validate_overview_headline(headline_text)

                groups_request = self._prompt_builder.build_overview_groups_text_request(
                    articles,
                    headline=headline,
                )
                groups_text = await self._request_chat(groups_request, target=target)
                raw_lines = self._outputs.coerce_summary_lines(groups_text)
                if not raw_lines:
                    raise ValueError("plain text overview groups are empty")

                normalized = [str(item or "").strip() for item in raw_lines if str(item or "").strip()]
                if not normalized:
                    raise ValueError("plain text overview groups are empty")

                normalized = self._outputs.sanitize_overview_lines(
                    normalized,
                    strip_first_line_refs=False,
                )
                groups = self._outputs.validate_overview_groups(normalized, valid_ids)
                return self._outputs.postprocess_overview_lines([headline, *groups], articles)
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

    async def _request_json(
        self,
        request: JSONRequestSpec,
        *,
        target: AIRetryTarget,
    ) -> dict[str, Any]:
        raw = await self._transport.request_json(
            request.system_prompt,
            request.user_prompt,
            request.schema,
            request.step,
            target=target,
        )
        if not isinstance(raw, dict):
            raise ValueError(f"{request.step} output is not a JSON object")
        return raw

    async def _request_chat(
        self,
        request: ChatRequestSpec,
        *,
        target: AIRetryTarget,
    ) -> str:
        return await self._transport.chat(request.messages, response_format=None, target=target)

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


__all__ = ["AIProcessingError", "AIProcessor"]
