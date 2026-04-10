from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .i18n import Locale
from .models import CleanedArticle, ProcessedArticle
from .utils import json_dumps


@dataclass(frozen=True)
class JSONRequestSpec:
    step: str
    system_prompt: str
    user_prompt: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class ChatRequestSpec:
    messages: list[dict[str, str]]


class AIPromptBuilder:
    def __init__(
        self,
        *,
        locale: Locale,
        preferred_categories: Sequence[str],
        summary_line_target_len: int,
    ) -> None:
        self._locale = locale
        self._preferred_categories = [str(item).strip() for item in preferred_categories if str(item).strip()]
        self._summary_line_target_len = max(int(summary_line_target_len), 1)

    def configured_preferred_categories(self) -> list[str]:
        return list(self._preferred_categories)

    def categorization_content(self, content: str) -> str:
        return self._truncate_prompt_content(content, max_chars=1200)

    def category_suggestion_content(self, content: str) -> str:
        return self._truncate_prompt_content(content, max_chars=400)

    def categorization_max_categories(self, article_count: int) -> int:
        if article_count <= 0:
            return 1
        dynamic_cap = round(1.5 * math.log2(max(article_count, 1)))
        bounded_cap = max(4, min(dynamic_cap, 10))
        return min(article_count, bounded_cap)

    def build_summarization_request(
        self,
        articles: Sequence[CleanedArticle],
    ) -> JSONRequestSpec:
        articles_json = json_dumps(
            [
                {
                    "id": article.id,
                    "title": article.title,
                    "content": article.content,
                }
                for article in articles
            ]
        )
        example_output = self._render_locale_text("prompts.summarization.example_output")
        return JSONRequestSpec(
            step="summarization",
            system_prompt=self._locale.render_prompt("summarization", "system"),
            user_prompt=self._locale.render_prompt(
                "summarization",
                "user",
                count=len(articles),
                example_output=example_output,
                articles_json=articles_json,
            ),
            schema={
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
            },
        )

    def build_categorization_request(
        self,
        articles: Sequence[CleanedArticle],
        *,
        category_candidates: Sequence[str],
        used_categories: Sequence[str],
        total_article_count: int,
    ) -> JSONRequestSpec:
        preferred_categories = self.configured_preferred_categories()
        suggested_categories = []
        if not preferred_categories:
            suggested_categories = [str(item).strip() for item in category_candidates if str(item).strip()]

        prompt_kwargs = self.build_categorization_prompt_kwargs(
            shard_article_count=len(articles),
            total_article_count=total_article_count,
            preferred_categories=preferred_categories,
            suggested_categories=suggested_categories,
            used_categories=used_categories,
            example_output=self._render_locale_text("prompts.categorization.example_output"),
            articles_json=json_dumps(
                [
                    {
                        "id": article.id,
                        "title": article.title,
                        "content": self.categorization_content(article.content),
                    }
                    for article in articles
                ]
            ),
        )
        return JSONRequestSpec(
            step="categorization",
            system_prompt=self._locale.render_prompt("categorization", "system", **prompt_kwargs),
            user_prompt=self._locale.render_prompt("categorization", "user", **prompt_kwargs),
            schema={
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
            },
        )

    def build_category_suggestion_request(
        self,
        articles: Sequence[CleanedArticle],
        *,
        existing_categories: Sequence[str],
        total_article_count: int,
    ) -> JSONRequestSpec:
        prompt_kwargs = self.build_category_suggestion_prompt_kwargs(
            shard_article_count=len(articles),
            total_article_count=total_article_count,
            existing_categories=existing_categories,
            example_output=self._render_locale_text("prompts.category_suggestion.example_output"),
            articles_json=json_dumps(
                [
                    {
                        "title": article.title,
                        "content": self.category_suggestion_content(article.content),
                    }
                    for article in articles
                ]
            ),
        )
        return JSONRequestSpec(
            step="category_suggestion",
            system_prompt=self._locale.render_prompt("category_suggestion", "system", **prompt_kwargs),
            user_prompt=self._locale.render_prompt("category_suggestion", "user", **prompt_kwargs),
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["categories"],
            },
        )

    def build_overview_headline_request(
        self,
        articles: Sequence[ProcessedArticle],
    ) -> JSONRequestSpec:
        article_lines = json_dumps(
            [
                {
                    "title": article.title,
                    "oneLine": article.one_line,
                }
                for article in articles
            ]
        )
        example_output = str(self._locale.require("prompts.overview_headline.example_output"))
        return JSONRequestSpec(
            step="overview_headline",
            system_prompt=self._locale.render_prompt(
                "overview_headline",
                "structured",
                summary_target=self._summary_line_target_len,
            ),
            user_prompt=self._locale.render_prompt(
                "overview_headline",
                "user",
                count=len(articles),
                article_lines=article_lines,
                example_output=example_output,
            ),
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                },
                "required": ["headline"],
            },
        )

    def build_overview_groups_request(
        self,
        articles: Sequence[ProcessedArticle],
        *,
        headline: str,
    ) -> JSONRequestSpec:
        article_lines = json_dumps(
            [
                {
                    "id": article.id,
                    "title": article.title,
                    "oneLine": article.one_line,
                }
                for article in articles
            ]
        )
        example_output = str(self._locale.require("prompts.overview_groups.example_output"))
        max_id = max((item.id for item in articles), default=0)
        return JSONRequestSpec(
            step="overview_groups",
            system_prompt=self._locale.render_prompt(
                "overview_groups",
                "structured",
                summary_target=self._summary_line_target_len,
            ),
            user_prompt=self._locale.render_prompt(
                "overview_groups",
                "user",
                count=len(articles),
                max_id=max_id,
                headline=headline,
                article_lines=article_lines,
                example_output=example_output,
            ),
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "groupLines": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["groupLines"],
            },
        )

    def build_overview_headline_text_request(
        self,
        articles: Sequence[ProcessedArticle],
    ) -> ChatRequestSpec:
        article_lines = json_dumps(
            [
                {
                    "title": article.title,
                    "oneLine": article.one_line,
                }
                for article in articles
            ]
        )
        example_output = str(self._locale.require("prompts.overview_headline.plain_text_example_output"))
        return ChatRequestSpec(
            messages=[
                {
                    "role": "system",
                    "content": self._locale.render_prompt(
                        "overview_headline",
                        "plain_text",
                        summary_target=self._summary_line_target_len,
                    ),
                },
                {
                    "role": "user",
                    "content": self._locale.render_prompt(
                        "overview_headline",
                        "user",
                        count=len(articles),
                        article_lines=article_lines,
                        example_output=example_output,
                    ),
                },
            ]
        )

    def build_overview_groups_text_request(
        self,
        articles: Sequence[ProcessedArticle],
        *,
        headline: str,
    ) -> ChatRequestSpec:
        article_lines = json_dumps(
            [
                {
                    "id": article.id,
                    "title": article.title,
                    "oneLine": article.one_line,
                }
                for article in articles
            ]
        )
        max_id = max((item.id for item in articles), default=0)
        example_output = str(self._locale.require("prompts.overview_groups.example_output"))
        return ChatRequestSpec(
            messages=[
                {
                    "role": "system",
                    "content": self._locale.render_prompt(
                        "overview_groups",
                        "plain_text",
                        summary_target=self._summary_line_target_len,
                    ),
                },
                {
                    "role": "user",
                    "content": self._locale.render_prompt(
                        "overview_groups",
                        "user",
                        count=len(articles),
                        max_id=max_id,
                        headline=headline,
                        article_lines=article_lines,
                        example_output=example_output,
                    ),
                },
            ]
        )

    def build_categorization_prompt_kwargs(
        self,
        *,
        shard_article_count: int,
        total_article_count: int,
        preferred_categories: Sequence[str],
        suggested_categories: Sequence[str],
        used_categories: Sequence[str],
        example_output: str,
        articles_json: str,
    ) -> dict[str, Any]:
        preferred_categories = [str(item).strip() for item in preferred_categories if str(item).strip()]
        suggested_categories = [str(item).strip() for item in suggested_categories if str(item).strip()]
        used_categories = [str(item).strip() for item in used_categories if str(item).strip()]

        category_candidates_name = ""
        category_candidates: list[str] = []
        if preferred_categories:
            category_candidates_name = "preferred_categories"
            category_candidates = preferred_categories
        elif suggested_categories:
            category_candidates_name = "suggested_categories"
            category_candidates = suggested_categories

        category_candidates_block = ""
        if category_candidates:
            category_candidates_block = self._render_locale_text(
                "prompts.categorization.blocks.category_candidates",
                category_candidates_name=category_candidates_name,
                category_candidates_json=json_dumps(category_candidates),
            )

        used_instruction = ""
        used_block = ""
        if used_categories:
            used_instruction = self._render_locale_text(
                "prompts.categorization.instructions.already_used_categories",
            )
            used_block = self._render_locale_text(
                "prompts.categorization.blocks.already_used_categories",
                used_categories_json=json_dumps(used_categories),
            )

        max_categories = self.categorization_max_categories(total_article_count)
        return {
            "count": shard_article_count,
            "max_categories": max_categories,
            "category_language_instruction": self._render_locale_text(
                "prompts.categorization.category_language_instruction"
            ),
            "already_used_categories_instruction": used_instruction,
            "categorization_policy_instruction": self._categorization_policy_instruction(
                max_categories=max_categories,
                category_candidates_name=category_candidates_name,
                has_category_candidates=bool(category_candidates),
                has_used_categories=bool(used_categories),
            ),
            "category_candidates_block": category_candidates_block,
            "already_used_categories_block": used_block,
            "example_output": example_output,
            "articles_json": articles_json,
        }

    def build_category_suggestion_prompt_kwargs(
        self,
        *,
        shard_article_count: int,
        total_article_count: int,
        existing_categories: Sequence[str],
        example_output: str,
        articles_json: str,
    ) -> dict[str, Any]:
        existing_categories = [str(item).strip() for item in existing_categories if str(item).strip()]

        existing_instruction = ""
        existing_block = ""
        if existing_categories:
            existing_instruction = self._render_locale_text(
                "prompts.category_suggestion.instructions.existing_categories",
            )
            existing_block = self._render_locale_text(
                "prompts.category_suggestion.blocks.existing_categories",
                existing_categories_json=json_dumps(existing_categories),
            )

        max_categories = self.categorization_max_categories(total_article_count)
        return {
            "count": shard_article_count,
            "max_categories": max_categories,
            "category_language_instruction": self._render_locale_text(
                "prompts.category_suggestion.category_language_instruction"
            ),
            "existing_categories_instruction": existing_instruction,
            "category_suggestion_policy_instruction": self._category_suggestion_policy_instruction(
                max_categories=max_categories,
                has_existing_categories=bool(existing_categories),
            ),
            "existing_categories_block": existing_block,
            "example_output": example_output,
            "articles_json": articles_json,
        }

    def _truncate_prompt_content(self, text: str, *, max_chars: int) -> str:
        content = str(text or "").strip()
        if max_chars <= 0 or len(content) <= max_chars:
            return content
        head_chars = max(max_chars // 2, 1)
        tail_chars = max(max_chars - head_chars - 1, 0)
        if tail_chars <= 0:
            return content[:max_chars]
        return f"{content[:head_chars].rstrip()}…{content[-tail_chars:].lstrip()}"

    def _category_suggestion_policy_instruction(
        self,
        *,
        max_categories: int,
        has_existing_categories: bool,
    ) -> str:
        key = (
            "prompts.category_suggestion.policies.with_existing"
            if has_existing_categories
            else "prompts.category_suggestion.policies.without_existing"
        )
        return self._render_locale_text(key, max_categories=max_categories)

    def _categorization_policy_instruction(
        self,
        *,
        max_categories: int,
        category_candidates_name: str,
        has_category_candidates: bool,
        has_used_categories: bool,
    ) -> str:
        if has_category_candidates:
            key = (
                "prompts.categorization.policies.with_candidates_and_used"
                if has_used_categories
                else "prompts.categorization.policies.with_candidates"
            )
            return self._render_locale_text(
                key,
                category_candidates_name=category_candidates_name,
                max_categories=max_categories,
            )
        return self._render_locale_text(
            "prompts.categorization.policies.without_candidates",
            max_categories=max_categories,
        )

    def _render_locale_text(self, dotted_key: str, **kwargs: Any) -> str:
        template = str(self._locale.require(dotted_key))
        return template.format(**kwargs)


__all__ = [
    "AIPromptBuilder",
    "ChatRequestSpec",
    "JSONRequestSpec",
]
