from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import unicodedata
from typing import Any, Awaitable, Callable, Dict, List, Sequence, Tuple, TypeVar

import tiktoken

from .ai_debug import AIDebugSink
from .ai_transport import AITransport
from .ai_processor_types import AIProcessingError
from .config import AIConfig, EnvConfig
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

        self.encoding = None
        try:
            self.encoding = tiktoken.encoding_for_model(env_config.openai_model)
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

        if estimated_tokens < threshold:
            async with self._timer.async_stage("  Phase 1"):
                phase1 = await self._phase1_for_batch(articles, shard_label="Shard 1")
        else:
            with self._timer.stage("  Sharding"):
                shards = self._build_shards(articles)
            self.logger.info("AI shard mode: shard_count=%d", len(shards))
            async with self._timer.async_stage("  Phase 1"):
                phase1 = await self._phase1_for_shards(shards)

        with self._timer.stage("  Build articles"):
            processed_articles = self._build_processed_articles(articles, phase1)
        try:
            async with self._timer.async_stage("  Phase 2"):
                summary_lines = await self._phase2_overview(processed_articles)
        except AIProcessingError as exc:
            if self.cfg.phase2_text_fallback:
                self.logger.warning("Phase 2 JSON overview failed, trying text fallback: %s", exc)
                try:
                    summary_lines = await self._phase2_overview_text_fallback(processed_articles)
                except AIProcessingError as fallback_exc:
                    if self.cfg.phase2_local_fallback:
                        self.logger.warning(
                            "Phase 2 text fallback failed, using local rule-based overview: %s",
                            fallback_exc,
                        )
                        summary_lines = self._build_local_overview(
                            processed_articles,
                            locale=self.locale,
                        )
                    else:
                        raise AIProcessingError(
                            f"Phase 2 text fallback failed (strict mode, local fallback disabled): {fallback_exc}"
                        ) from fallback_exc
            elif self.cfg.phase2_local_fallback:
                self.logger.warning("Phase 2 failed, using local rule-based overview: %s", exc)
                summary_lines = self._build_local_overview(processed_articles, locale=self.locale)
            else:
                raise AIProcessingError(f"Phase 2 failed (strict mode): {exc}") from exc
        with self._timer.stage("  Categories"):
            categories = self._collect_categories(processed_articles, locale=self.locale)

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
        default_category = self._default_category(self.locale)
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
                category=default_category,
            )
            for item in articles
        ]
        return ProcessedResult(
            articles=degraded_articles,
            categories=[default_category] if degraded_articles else [],
            summary_lines=[self._locale_fallback_text("phase2_failed", "AI overview generation failed. Original article list shown below.")],
            degraded=True,
            warnings=[resolved_warning_text] if resolved_warning_text else [],
        )

    def _locale_fallback_text(self, key: str, default: str) -> str:
        text = str(self.locale.fallback_texts.get(key, "") or "").strip()
        return text or default

    async def aclose(self) -> None:
        await self._transport.aclose()

    def _dump_debug(self, event: str, payload: Dict[str, Any], force: bool = False) -> None:
        self._debug.dump(event, payload, force=force)

    async def _run_json_phase(
        self,
        *,
        phase: str,
        log_label: str,
        request_fn: Callable[[], Awaitable[Any]],
        validate_fn: Callable[[Any], _T],
        timer_api_label: str,
        timer_validate_label: str,
    ) -> _T:
        schema_attempt = 0
        transient_attempt = 0
        api_time = 0.0
        val_time = 0.0

        try:
            while True:
                data: object = None
                t0 = time.monotonic()
                try:
                    data = await request_fn()
                    api_time += time.monotonic() - t0

                    t0 = time.monotonic()
                    result = validate_fn(data)
                    val_time += time.monotonic() - t0
                    return result
                except (json.JSONDecodeError, ValueError) as exc:
                    val_time += time.monotonic() - t0
                    schema_attempt += 1
                    self._dump_debug(
                        f"{phase}.schema_or_validation_error",
                        {
                            "attempt": schema_attempt,
                            "error": str(exc),
                            "response_data": data,
                        },
                        force=True,
                    )
                    if schema_attempt <= self.cfg.schema_retry_max:
                        self.logger.warning(
                            "%s schema/validation failed, retrying: attempt=%d, error=%s",
                            log_label,
                            schema_attempt,
                            exc,
                        )
                        continue
                    raise AIProcessingError(f"{phase} schema/validation failed: {exc}") from exc
                except Exception as exc:
                    self._dump_debug(
                        f"{phase}.call_error",
                        {"error": str(exc), "response_data": data},
                        force=True,
                    )
                    if self._is_transient_error(exc) and transient_attempt < self.cfg.transient_retry_max:
                        delay = self._backoff_delay(transient_attempt)
                        transient_attempt += 1
                        self.logger.warning(
                            "%s transient error, backing off: attempt=%d, delay=%.2fs, error=%s",
                            log_label,
                            transient_attempt,
                            delay,
                            exc,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise AIProcessingError(f"{phase} call failed: {exc}") from exc
        finally:
            self._timer.record(timer_api_label, api_time)
            self._timer.record(timer_validate_label, val_time)

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
                "source": a.source,
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
            article_chars = len(article.title) + len(article.content) + len(article.source)
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

    async def _phase1_for_shards(
        self, shards: Sequence[Sequence[CleanedArticle]]
    ) -> Dict[int, Dict[str, str]]:
        async def run_single(
            shard: Sequence[CleanedArticle], label: str
        ) -> Dict[int, Dict[str, str]]:
            async with self._semaphore:
                async with self._timer.async_stage(f"    {label}"):
                    return await self._phase1_for_batch(list(shard), shard_label=label)

        results = await asyncio.gather(
            *(run_single(s, f"Shard {i+1}") for i, s in enumerate(shards))
        )
        merged: Dict[int, Dict[str, str]] = {}
        for item in results:
            merged.update(item)
        return merged

    async def _phase1_for_batch(
        self, articles: List[CleanedArticle], *, shard_label: str = "",
    ) -> Dict[int, Dict[str, str]]:
        expected_ids = {a.id for a in articles}
        title_by_id = {a.id: a.title for a in articles}
        prefix = f"      {shard_label} " if shard_label else "    "
        return await self._run_json_phase(
            phase="phase1",
            log_label="Phase 1",
            request_fn=lambda: self._request_phase1(articles, locale=self.locale),
            validate_fn=lambda data: self._normalize_phase1_result(
                data,
                expected_ids=expected_ids,
                title_by_id=title_by_id,
            ),
            timer_api_label=f"{prefix}API",
            timer_validate_label=f"{prefix}Validate",
        )

    async def _phase2_overview(self, articles: List[ProcessedArticle]) -> List[str]:
        valid_ids = {a.id for a in articles}
        return await self._run_json_phase(
            phase="phase2",
            log_label="Phase 2",
            request_fn=lambda: self._request_phase2(articles, locale=self.locale),
            validate_fn=lambda data: self._postprocess_phase2_lines(
                self._validate_phase2(data, valid_ids, locale=self.locale),
                articles,
                locale=self.locale,
            ),
            timer_api_label="    API",
            timer_validate_label="    Validate",
        )

    async def _phase2_overview_text_fallback(
        self, articles: List[ProcessedArticle]
    ) -> List[str]:
        valid_ids = {a.id for a in articles}
        transient_attempt = 0

        while True:
            text = ""
            try:
                text = await self._request_phase2_text(articles, locale=self.locale)
                raw_lines = self._coerce_summary_lines(text)
                if not raw_lines:
                    raise ValueError("plain text overview is empty")

                normalized = [str(x or "").strip() for x in raw_lines if str(x or "").strip()]
                if not normalized:
                    raise ValueError("plain text overview is empty")

                normalized = self._sanitize_phase2_lines(normalized)
                normalized = self._normalize_phase2_line_count(
                    normalized,
                    articles,
                    locale=self.locale,
                )
                self._validate_phase2(normalized, valid_ids, locale=self.locale)
                return self._postprocess_phase2_lines(
                    normalized,
                    articles,
                    locale=self.locale,
                )
            except Exception as exc:
                self._dump_debug(
                    "phase2.text_fallback_error",
                    {"error": str(exc), "response_text": text},
                    force=True,
                )
                if self._is_transient_error(exc) and transient_attempt < self.cfg.transient_retry_max:
                    delay = self._backoff_delay(transient_attempt)
                    transient_attempt += 1
                    self.logger.warning(
                        "Phase 2 text fallback transient error, backing off: attempt=%d, delay=%.2fs, error=%s",
                        transient_attempt,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AIProcessingError(f"Phase 2 text fallback failed: {exc}") from exc

    def _validate_phase1(
        self,
        data: Dict[str, Any],
        expected_ids: set[int],
        title_by_id: Dict[int, str],
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

            if "id" not in item or "oneLine" not in item or "category" not in item:
                raise ValueError("perArticle element missing required fields")

            article_id = int(item["id"])
            one_line = str(item["oneLine"] or "").strip()
            category = str(item["category"] or "").strip() or self._default_category(locale)

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
            mismatch = self._find_phase1_alignment_mismatch(
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
            rows.append({"id": article_id, "oneLine": one_line, "category": category})

        if seen_ids != expected_ids:
            missing = sorted(expected_ids - seen_ids)
            extra = sorted(seen_ids - expected_ids)
            raise ValueError(f"id set mismatch: missing={missing}, extra={extra}")

        return rows

    def _normalize_phase1_result(
        self,
        data: Dict[str, Any],
        *,
        expected_ids: set[int],
        title_by_id: Dict[int, str],
    ) -> Dict[int, Dict[str, str]]:
        parsed = self._validate_phase1(
            data,
            expected_ids,
            title_by_id,
            locale=self.locale,
        )
        normalized: Dict[int, Dict[str, str]] = {}
        for entry in parsed:
            normalized[entry["id"]] = {
                "one_line": entry["oneLine"],
                "category": self._normalize_category(entry["category"], locale=self.locale),
            }
        return normalized

    def _default_category(self, locale: Locale) -> str:
        return locale.default_category

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

    def _find_phase1_alignment_mismatch(
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

    def _normalize_category(self, category: str, locale: Locale) -> str:
        category = (category or "").strip()
        return category or self._default_category(locale)

    def _collect_categories(
        self,
        articles: Sequence[ProcessedArticle],
        locale: Locale,
    ) -> List[str]:
        default_category = self._default_category(locale)
        present_order: List[str] = []
        present_set: set[str] = set()
        for item in articles:
            cat = (item.category or "").strip() or default_category
            if cat in present_set:
                continue
            present_set.add(cat)
            present_order.append(cat)

        if not present_order and articles:
            return [default_category]

        taxonomy = locale.taxonomy or self.cfg.taxonomy
        ordered = [c for c in taxonomy if c in present_set]
        for cat in present_order:
            if cat not in ordered:
                ordered.append(cat)
        return ordered

    def _build_processed_articles(
        self, cleaned: Sequence[CleanedArticle], phase1_map: Dict[int, Dict[str, str]]
    ) -> List[ProcessedArticle]:
        items: List[ProcessedArticle] = []
        for article in cleaned:
            mapped = phase1_map.get(article.id)
            if not mapped:
                raise AIProcessingError(f"Phase 1 result missing id={article.id}")
            items.append(
                ProcessedArticle(
                    id=article.id,
                    title=article.title,
                    link=article.link,
                    pub_date=article.pub_date,
                    source=article.source,
                    one_line=mapped["one_line"],
                    category=mapped["category"],
                )
            )
        return items

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

    def _backoff_delay(self, attempt: int) -> float:
        idx = min(attempt, max(len(self.cfg.backoff_seconds) - 1, 0))
        base = float(self.cfg.backoff_seconds[idx]) if self.cfg.backoff_seconds else 1.0
        jitter = random.randint(0, max(self.cfg.jitter_ms_max, 0)) / 1000.0
        return base + jitter

    def _validate_phase2(
        self,
        data: Any,
        valid_ids: set[int],
        locale: Locale,
    ) -> List[str]:
        lines = self._coerce_summary_lines(data)
        if not isinstance(lines, list):
            raise ValueError("summaryLines must be an array")

        cleaned = [str(x or "").strip() for x in lines if str(x or "").strip()]
        if not (5 <= len(cleaned) <= 9):
            raise ValueError(f"summaryLines must have 5-9 lines, got {len(cleaned)}")
        if self._is_placeholder_overview_line(cleaned[0], locale=locale):
            raise ValueError("summaryLines first line is a placeholder, missing valid overview content")

        normalized: List[str] = []
        ref_re = re.compile(r"\[(\d+)\]")
        for idx, line in enumerate(cleaned, start=1):
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
            normalized.append(adjusted)
            for ref in ref_re.findall(adjusted):
                if int(ref) not in valid_ids:
                    raise ValueError(f"overview references non-existent article id: {ref}")
        return normalized

    def _is_placeholder_overview_line(
        self,
        text: str,
        locale: Locale,
    ) -> bool:
        compact = re.sub(r"\s+", "", str(text or ""))
        compact = re.sub(r"[：:，,。.!?！？；;、\-_（）()\[\]【】]", "", compact)
        return compact in locale.overview_placeholders

    def _postprocess_phase2_lines(
        self,
        lines: List[str],
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> List[str]:
        if not lines:
            return self._build_local_overview(articles, locale=locale)

        out = [str(x or "").strip() for x in lines if str(x or "").strip()]
        if not out:
            return self._build_local_overview(articles, locale=locale)

        ref_re = re.compile(r"\[(\d+)\]")
        if ref_re.search(out[0]):
            out[0] = self._build_macro_overview_line(articles, locale=locale)

        detail = out[1:]
        if detail:
            multi_ref_count = sum(1 for line in detail if len(ref_re.findall(line)) >= 2)
            min_multi = max(2, len(detail) // 2)
            if multi_ref_count < min_multi:
                self.logger.warning(
                    "Phase 2 style does not meet aggregation requirements (insufficient multi-ref lines), using local overview"
                )
                if self.cfg.phase2_local_fallback:
                    local = self._build_local_overview(articles, locale=locale)
                    ai_first = self._sanitize_phase2_lines([out[0]]) if out else []
                    if ai_first:
                        first_line = ai_first[0]
                        if first_line and not ref_re.search(first_line):
                            local[0] = first_line
                    return local
                raise ValueError("Phase 2 style does not meet aggregation requirements (insufficient multi-ref lines)")

        if len(out) < 5:
            if self.cfg.phase2_local_fallback:
                local_lines = self._build_local_overview(articles, locale=locale)
                out = [out[0]] + local_lines[1:]
            else:
                raise ValueError("Phase 2 overview has too few lines (strict mode)")
        return out[:9]

    def _sanitize_phase2_lines(self, lines: List[str]) -> List[str]:
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

            if idx == 0:
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

    def _normalize_phase2_line_count(
        self,
        lines: List[str],
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> List[str]:
        out = [str(x or "").strip() for x in lines if str(x or "").strip()]
        if not out:
            return (
                self._build_local_overview(articles, locale=locale)
                if self.cfg.phase2_local_fallback
                else []
            )
        if len(out) > 9:
            return out[:9]
        if len(out) >= 5:
            return out

        if not self.cfg.phase2_local_fallback:
            return out

        local_lines = self._build_local_overview(articles, locale=locale)
        index = 1
        while len(out) < 5 and index < len(local_lines):
            out.append(local_lines[index])
            index += 1
        return out[:9]

    def _phase2_fallback_text(
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
            return [self._phase2_fallback_text("no_overview", "No overview available today.", locale)]

        first_line = self._build_macro_overview_line(articles, locale=locale)
        by_category: Dict[str, List[ProcessedArticle]] = {}
        default_category = self._default_category(locale)
        for item in articles:
            category = (item.category or "").strip() or default_category
            by_category.setdefault(category, []).append(item)

        ordered_categories = sorted(
            by_category.keys(), key=lambda x: len(by_category.get(x, [])), reverse=True
        )

        used_ids: set[int] = set()
        detail_lines: List[str] = []

        for cat in ordered_categories:
            candidates = [x for x in by_category.get(cat, []) if x.id not in used_ids]
            if len(candidates) < 2:
                continue
            line, ids = self._build_grouped_line(candidates, min_items=2, max_items=2)
            if not line:
                continue
            detail_lines.append(line)
            used_ids.update(ids)
            if len(detail_lines) >= 8:
                break

        if len(detail_lines) < 8:
            remaining = [x for x in articles if x.id not in used_ids]
            idx = 0
            while idx < len(remaining) and len(detail_lines) < 8:
                chunk = remaining[idx : idx + 2]
                if not chunk:
                    break
                line, ids = self._build_grouped_line(chunk, min_items=1, max_items=2)
                if line:
                    detail_lines.append(line)
                    used_ids.update(ids)
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
        counter: Dict[str, int] = {}
        default_category = self._default_category(locale)
        for item in articles:
            category = (item.category or "").strip() or default_category
            counter[category] = counter.get(category, 0) + 1
        top_categories = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:3]
        is_english = locale.lang.lower().startswith("en")
        separator = ", " if is_english else "、"
        cat_text = separator.join([name for name, _ in top_categories])
        if not cat_text:
            cat_text = "multiple sectors" if is_english else "多领域"
        theme_counts = self._extract_theme_counts(articles, locale=locale)
        top_themes = [
            name for name, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True) if count > 0
        ][:2]
        if top_themes:
            theme_text = separator.join(top_themes)
            if is_english:
                line = (
                    f"Today's key themes center on {theme_text}, with {cat_text} developments "
                    "appearing in quick succession and sentiment moving alongside risk appetite."
                )
            else:
                line = f"今日主线集中在{theme_text}，{cat_text}相关事件密集出现，市场情绪与风险偏好同步波动。"
        else:
            if is_english:
                line = (
                    f"Today's key themes center on {cat_text}, as cross-market developments "
                    "interact and structural opportunities coexist with volatility."
                )
            else:
                line = f"今日主线集中在{cat_text}，跨市场消息交叉影响，结构性机会与波动并存。"
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

    async def _request_phase1(
        self,
        articles: List[CleanedArticle],
        locale: Locale,
    ) -> Dict[str, Any]:
        taxonomy = locale.taxonomy or self.cfg.taxonomy
        taxonomy_separator = ", " if locale.lang.lower().startswith("en") else "、"
        taxonomy_text = taxonomy_separator.join(taxonomy)
        taxonomy_json = json_dumps(taxonomy)
        user_payload = [
            {
                "id": x.id,
                "title": x.title,
                "content": x.content,
                "source": x.source,
            }
            for x in articles
        ]
        articles_json = json_dumps(user_payload)
        example_output = '{\n  "perArticle":[{"id":1,"oneLine":"...","category":"..."}]\n}'
        system_prompt = locale.render_prompt(
            "phase1",
            "system",
            taxonomy_text=taxonomy_text,
            taxonomy_json=taxonomy_json,
        )
        user_prompt = locale.render_prompt(
            "phase1",
            "user",
            taxonomy_text=taxonomy_text,
            count=len(articles),
            taxonomy_json=taxonomy_json,
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
                            "category": {"type": "string"},
                        },
                        "required": ["id", "oneLine", "category"],
                    },
                },
            },
            "required": ["perArticle"],
        }

        raw = await self._transport.request_json(system_prompt, user_prompt, schema, "phase1")
        if not isinstance(raw, dict):
            raise ValueError("Phase 1 output is not a JSON object")
        return raw

    async def _request_phase2(
        self,
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> Any:
        summary_target = int(self._summary_line_target_len)
        user_payload = [
            {
                "id": x.id,
                "title": x.title,
                "oneLine": x.one_line,
                "category": x.category,
                "source": x.source,
            }
            for x in articles
        ]
        article_lines = json_dumps(user_payload)
        example_output = locale.get("prompts.phase2.example_output", '{"summaryLines":["..."]}')
        max_id = max((item.id for item in articles), default=0)
        system_prompt = locale.render_prompt(
            "phase2",
            "structured",
            summary_target=summary_target,
        )
        user_prompt = locale.render_prompt(
            "phase2",
            "user",
            count=len(articles),
            max_id=max_id,
            article_lines=article_lines,
            example_output=example_output,
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summaryLines": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["summaryLines"],
        }
        return await self._transport.request_json(system_prompt, user_prompt, schema, "phase2")

    async def _request_phase2_text(
        self,
        articles: List[ProcessedArticle],
        locale: Locale,
    ) -> str:
        summary_target = int(self._summary_line_target_len)
        user_payload = [
            {
                "id": x.id,
                "title": x.title,
                "oneLine": x.one_line,
                "category": x.category,
                "source": x.source,
            }
            for x in articles
        ]
        article_lines = json_dumps(user_payload)
        example_output = locale.get("prompts.phase2.example_output", '{"summaryLines":["..."]}')
        system_prompt = locale.render_prompt(
            "phase2",
            "plain_text",
            summary_target=summary_target,
        )
        user_prompt = locale.render_prompt(
            "phase2",
            "user",
            count=len(articles),
            max_id=max((item.id for item in articles), default=0),
            article_lines=article_lines,
            example_output=example_output,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._transport.chat(messages, response_format=None)


__all__ = ["AIProcessingError", "AIProcessor"]
