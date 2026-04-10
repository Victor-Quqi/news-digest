from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Callable, Sequence

from .ai_processor_types import AIProcessingError
from .i18n import Locale
from .models import CleanedArticle, ProcessedArticle


class AIOutputProcessor:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        locale: Locale,
        preferred_categories: Sequence[str],
        one_line_hard_units: float,
        one_line_soft_units: float,
        one_line_trim_target_units: float,
        summary_line_target_len: int,
        summary_line_hard_limit: float,
        summary_line_soft_limit: float,
    ) -> None:
        self._logger = logger
        self._locale = locale
        self._preferred_categories = [str(item).strip() for item in preferred_categories if str(item).strip()]
        self._one_line_hard_units = one_line_hard_units
        self._one_line_soft_units = one_line_soft_units
        self._one_line_trim_target_units = one_line_trim_target_units
        self._summary_line_target_len = summary_line_target_len
        self._summary_line_hard_limit = summary_line_hard_limit
        self._summary_line_soft_limit = summary_line_soft_limit

    def normalize_summarization_result(
        self,
        data: dict[str, Any],
        *,
        expected_ids: set[int],
        title_by_id: dict[int, str],
    ) -> dict[int, dict[str, str]]:
        parsed = self.validate_summarization(
            data,
            expected_ids=expected_ids,
            title_by_id=title_by_id,
        )
        return {
            entry["id"]: {"one_line": entry["oneLine"]}
            for entry in parsed
        }

    def normalize_categorization_result(
        self,
        data: dict[str, Any],
        *,
        expected_ids: set[int],
    ) -> dict[int, str]:
        parsed = self.validate_categorization(data, expected_ids=expected_ids)
        return {entry["id"]: entry["category"] for entry in parsed}

    def normalize_category_suggestion_result(
        self,
        data: dict[str, Any],
        *,
        max_categories: int,
    ) -> list[str]:
        return self.validate_category_suggestion(data, max_categories=max_categories)

    def validate_summarization(
        self,
        data: dict[str, Any],
        *,
        expected_ids: set[int],
        title_by_id: dict[int, str],
    ) -> list[dict[str, Any]]:
        per_article = data.get("perArticle")
        if not isinstance(per_article, list):
            raise ValueError("perArticle must be an array")

        rows: list[dict[str, Any]] = []
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
            mismatch = self.find_summarization_alignment_mismatch(
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

    def validate_categorization(
        self,
        data: dict[str, Any],
        *,
        expected_ids: set[int],
    ) -> list[dict[str, Any]]:
        per_article = data.get("perArticle")
        if not isinstance(per_article, list):
            raise ValueError("perArticle must be an array")

        rows: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in per_article:
            if not isinstance(item, dict):
                raise ValueError("perArticle element must be an object")
            if "id" not in item or "category" not in item:
                raise ValueError("perArticle element missing required fields")

            article_id = int(item["id"])
            category = self.normalize_category(str(item["category"] or ""))
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

    def validate_category_suggestion(
        self,
        data: dict[str, Any],
        *,
        max_categories: int,
    ) -> list[str]:
        categories = data.get("categories")
        if not isinstance(categories, list):
            raise ValueError("categories must be an array")

        normalized: list[str] = []
        seen: set[str] = set()
        for item in categories:
            category = self.normalize_category(str(item or ""))
            if not category:
                raise ValueError("suggested category must not be empty")
            if category in seen:
                continue
            seen.add(category)
            normalized.append(category)

        if not normalized:
            raise ValueError("categories must not be empty")
        if len(normalized) > max_categories:
            raise ValueError(f"categories must have 1-{max_categories} items, got {len(normalized)}")
        return normalized

    def validate_overview(
        self,
        data: Any,
        valid_ids: set[int],
    ) -> list[str]:
        lines = self.coerce_summary_lines(data)
        if not isinstance(lines, list):
            raise ValueError("summaryLines must be an array")

        cleaned = [str(item or "").strip() for item in lines if str(item or "").strip()]
        if not (4 <= len(cleaned) <= 9):
            raise ValueError(f"summaryLines must have 4-9 lines, got {len(cleaned)}")
        headline = self.validate_overview_headline(cleaned[0])
        groups = self.validate_overview_groups(cleaned[1:], valid_ids)
        return [headline, *groups]

    def validate_overview_headline(self, data: Any) -> str:
        headline = data.get("headline") if isinstance(data, dict) else data
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

    def validate_overview_groups(
        self,
        data: Any,
        valid_ids: set[int],
    ) -> list[str]:
        lines = self.coerce_summary_lines(data)
        if not isinstance(lines, list):
            raise ValueError("summaryLines must be an array")

        cleaned = [str(item or "").strip() for item in lines if str(item or "").strip()]
        if not (3 <= len(cleaned) <= 8):
            raise ValueError(f"summaryLines must have 3-8 lines, got {len(cleaned)}")

        normalized: list[str] = []
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

    def postprocess_overview_lines(
        self,
        lines: list[str],
        articles: list[ProcessedArticle],
    ) -> list[str]:
        out = [str(item or "").strip() for item in lines if str(item or "").strip()]
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

    def sanitize_overview_lines(
        self,
        lines: list[str],
        *,
        strip_first_line_refs: bool,
    ) -> list[str]:
        cleaned: list[str] = []
        ref_re = re.compile(r"\[(\d+)\]")
        artifact_re = re.compile(
            r"(?i)(websearch|web_search|search_with_snippets|tool_call|function_call|code_execution|\"query\"\\s*:)"
        )
        for idx, raw in enumerate(lines):
            line = re.sub(r"^\s*[\-\*\d\.\)\(、:：]+\s*", "", str(raw or "").strip())
            line = re.sub(r"</?(think|analysis|reasoning|thought)[^>]*>", "", line, flags=re.IGNORECASE).strip()
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

            refs = "".join(f"[{ref}]" for ref in ref_re.findall(line))
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
            ).strip()
            if merged:
                cleaned.append(merged)
        return cleaned

    def build_local_overview(self, articles: list[ProcessedArticle]) -> list[str]:
        if not articles:
            return [self._overview_fallback_text("no_overview", "No overview available today.")]

        first_line = self._build_macro_overview_line(articles)
        detail_lines: list[str] = []
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

    def build_processed_articles(
        self,
        cleaned: Sequence[CleanedArticle],
        summary_map: dict[int, dict[str, str]],
    ) -> list[ProcessedArticle]:
        items: list[ProcessedArticle] = []
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

    def apply_category_map(
        self,
        articles: Sequence[ProcessedArticle],
        category_map: dict[int, str],
    ) -> None:
        for article in articles:
            category = category_map.get(article.id)
            if category is None:
                raise AIProcessingError(f"Categorization result missing id={article.id}")
            article.category = category

    def collect_categories(
        self,
        articles: Sequence[ProcessedArticle],
    ) -> list[str]:
        present_order: list[str] = []
        present_set: set[str] = set()
        for item in articles:
            category = (item.category or "").strip()
            if not category or category in present_set:
                continue
            present_set.add(category)
            present_order.append(category)
        preferred = [category for category in self._preferred_categories if category in present_set]
        return preferred + [category for category in present_order if category not in preferred]

    def normalize_category(self, category: str) -> str:
        return (category or "").strip()

    def find_summarization_alignment_mismatch(
        self,
        *,
        article_id: int,
        one_line: str,
        title_by_id: dict[int, str],
    ) -> tuple[int, float, float] | None:
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

        if best_id != article_id and best_score >= 0.45 and best_score - own_score >= 0.25:
            return (best_id, own_score, best_score)
        return None

    def coerce_summary_lines(self, data: Any, depth: int = 0) -> list[str] | None:
        if depth > 4:
            return None

        if isinstance(data, list):
            return [str(item or "").strip() for item in data if str(item or "").strip()]

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
                    lines = self.coerce_summary_lines(data.get(key), depth + 1)
                    if lines:
                        return lines

            content = self._extract_chat_content_from_dict(data)
            if content:
                lines = self.coerce_summary_lines(content, depth + 1)
                if lines:
                    return lines
            return None

        if isinstance(data, str):
            text = data.strip()
            if not text:
                return None

            parsed = self._try_parse_json_payload(text)
            if parsed is not None and parsed is not data:
                lines = self.coerce_summary_lines(parsed, depth + 1)
                if lines:
                    return lines
            return [item.strip() for item in text.splitlines() if item.strip()]
        return None

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
            self._logger.warning(
                "%s too long (%.1f), trimmed to %.1f",
                context,
                value,
                float(value_fn(shortened)),
            )
            return shortened
        if value > hard_limit:
            self._logger.info("%s slightly over limit (%.1f), keeping original", context, value)
        return text

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
                for idx in range(len(raw) - 1):
                    bigram = raw[idx : idx + 2]
                    if bigram not in stopwords:
                        tokens.add(bigram)
            elif raw not in stopwords:
                tokens.add(raw)
        return tokens

    def _title_overlap_score(self, summary: str, title: str) -> float:
        summary_tokens = self._match_tokens(summary)
        title_tokens = self._match_tokens(title)
        if not summary_tokens or not title_tokens:
            return 0.0
        return len(summary_tokens & title_tokens) / max(len(summary_tokens), 1)

    def _char_units(self, ch: str) -> float:
        if not ch or ch.isspace():
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
        out: list[str] = []
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

    def _overview_fallback_text(self, key: str, default: str) -> str:
        text = str(self._locale.fallback_texts.get(key, "") or "").strip()
        return text or default

    def _build_macro_overview_line(self, articles: list[ProcessedArticle]) -> str:
        is_english = self._locale.lang.lower().startswith("en")
        separator = ", " if is_english else "、"
        theme_counts = self._extract_theme_counts(articles)
        top_themes = [name for name, count in sorted(theme_counts.items(), key=lambda item: item[1], reverse=True) if count > 0][:2]
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
        elif is_english:
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

    def _extract_theme_counts(self, articles: list[ProcessedArticle]) -> dict[str, int]:
        text = " ".join(
            self._normalize_summary_fragment(item.one_line or item.title)
            for item in articles
            if (item.one_line or item.title)
        )
        theme_keywords = self._locale.theme_keywords
        lower = text.lower()
        counts: dict[str, int] = {key: 0 for key in theme_keywords}
        for theme, keywords in theme_keywords.items():
            for keyword in keywords:
                counts[theme] += lower.count(keyword.lower())
        return counts

    def _normalize_summary_fragment(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\[(\d+)\]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", cleaned)
        return cleaned.rstrip("，,、；;：:。.!?！？ ")

    def _build_grouped_line(
        self,
        items: list[ProcessedArticle],
        min_items: int = 2,
        max_items: int = 3,
    ) -> tuple[str, list[int]]:
        chosen_parts: list[str] = []
        chosen_ids: list[int] = []
        limit = int(self._summary_line_target_len)

        for item in items[:max_items]:
            fragment = self._normalize_summary_fragment(item.one_line or item.title)
            if not fragment:
                continue

            trial_parts = chosen_parts + [fragment]
            trial_ids = chosen_ids + [item.id]
            refs = "".join(f"[{item_id}]" for item_id in trial_ids)
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

        refs = "".join(f"[{item_id}]" for item_id in chosen_ids)
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
            line = f"{self._condense_fragment(fragment, max_body_len)}[{item.id}]"
        return line

    def _trim_line_keep_refs(self, body: str, refs: str, max_len: int) -> str:
        max_body_len = max(max_len - len(refs), 0)
        return f"{self._condense_fragment(body or '', max_body_len)}{refs}"

    def _trim_summary_line_by_chars(self, text: str, target_len: float) -> str:
        max_len = max(int(target_len), 1)
        ref_re = re.compile(r"\[(\d+)\]")
        refs = "".join(f"[{ref}]" for ref in ref_re.findall(text or ""))
        body = ref_re.sub("", text or "").strip()
        if refs:
            return self._trim_line_keep_refs(body, refs, max_len)
        return self._condense_fragment(body, max_len)

    def _condense_fragment(self, text: str, max_len: int) -> str:
        cleaned = (text or "").strip()
        if max_len <= 0:
            return ""
        if len(cleaned) <= max_len:
            return cleaned

        parts = [part.strip() for part in re.split(r"[，；。,.!?！？:：]", cleaned) if part.strip()]
        for part in parts:
            if len(part) <= max_len:
                return part
        if max_len <= 4:
            return cleaned[:max_len]
        return cleaned[: max_len - 1].rstrip("，,、；;：:。.!?！？ ") + "…"

    def _try_parse_json_payload(self, text: str) -> Any | None:
        payload = (text or "").strip()
        if not payload:
            return None
        try:
            return json.loads(payload)
        except Exception:
            pass

        block = self._extract_first_json_block(payload)
        if not block:
            return None
        try:
            return json.loads(block)
        except Exception:
            return None

    def _extract_first_json_block(self, text: str) -> str:
        payload = (text or "").strip()
        if not payload:
            return ""

        fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", payload, flags=re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        start_positions = [idx for idx in (payload.find("{"), payload.find("[")) if idx >= 0]
        if not start_positions:
            return ""
        start = min(start_positions)
        opening = payload[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(payload)):
            ch = payload[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return payload[start : idx + 1]
        return ""

    def _extract_chat_content_from_dict(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""

        message = first.get("message")
        if isinstance(message, dict):
            joined = self._join_content_parts(message.get("content"))
            if joined:
                return joined

        delta = first.get("delta")
        if isinstance(delta, dict):
            joined = self._join_content_parts(delta.get("content"))
            if joined:
                return joined
        return ""

    def _join_content_parts(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    parts.append(stripped)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()


__all__ = ["AIOutputProcessor"]
