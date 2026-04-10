from __future__ import annotations

import unittest
from datetime import datetime

from src.ai_prompts import AIPromptBuilder
from src.i18n import Locale
from src.models import CleanedArticle


class MissingPromptLocale:
    lang = "zh"

    def render_prompt(self, step: str, role: str, **kwargs: object) -> str:
        raise KeyError(f"missing prompt: {step}.{role}")

    def require(self, dotted_key: str) -> str:
        raise KeyError(f"missing locale key: {dotted_key}")


class PromptBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.locale = Locale("zh")
        self.builder = AIPromptBuilder(
            locale=self.locale,
            preferred_categories=[],
            summary_line_target_len=120,
        )

    def test_dynamic_category_cap_matches_examples(self) -> None:
        self.assertEqual(self.builder.categorization_max_categories(20), 6)
        self.assertEqual(self.builder.categorization_max_categories(40), 8)
        self.assertEqual(self.builder.categorization_max_categories(100), 10)

    def test_categorization_prompt_omits_empty_optional_blocks(self) -> None:
        article = CleanedArticle(
            id=1,
            title="美国就业数据走强",
            link="https://example.com/1",
            pub_date=datetime(2026, 4, 10, 8, 0, 0),
            source="Reuters",
            content="就业与通胀数据共同影响美联储判断。",
        )

        request = self.builder.build_categorization_request(
            [article],
            category_candidates=[],
            used_categories=[],
            total_article_count=1,
        )

        self.assertNotIn("preferred_categories=", request.user_prompt)
        self.assertNotIn("suggested_categories=", request.user_prompt)
        self.assertNotIn("already_used_categories=", request.user_prompt)
        self.assertNotIn("already_used_categories 是前面分片已经用过的分类名", request.system_prompt)

    def test_prompt_kwargs_include_preferred_and_used_categories(self) -> None:
        builder = AIPromptBuilder(
            locale=self.locale,
            preferred_categories=["宏观政策", "科技AI"],
            summary_line_target_len=120,
        )

        kwargs = builder.build_categorization_prompt_kwargs(
            shard_article_count=20,
            total_article_count=20,
            preferred_categories=["宏观政策", "科技AI"],
            suggested_categories=[],
            used_categories=["宏观政策"],
            example_output='{"perArticle":[{"id":1,"category":"..."}]}',
            articles_json="[]",
        )

        self.assertIn("preferred_categories=", kwargs["category_candidates_block"])
        self.assertIn("already_used_categories=", kwargs["already_used_categories_block"])
        self.assertEqual(kwargs["max_categories"], 6)
        self.assertIn("优先直接使用 preferred_categories 里的分类名", kwargs["categorization_policy_instruction"])

    def test_prompt_kwargs_can_use_suggested_categories(self) -> None:
        kwargs = self.builder.build_categorization_prompt_kwargs(
            shard_article_count=20,
            total_article_count=20,
            preferred_categories=[],
            suggested_categories=["国际", "财经", "科技"],
            used_categories=["国际"],
            example_output='{"perArticle":[{"id":1,"category":"..."}]}',
            articles_json="[]",
        )

        self.assertIn("suggested_categories=", kwargs["category_candidates_block"])
        self.assertIn("优先直接使用 suggested_categories 里的分类名", kwargs["categorization_policy_instruction"])

    def test_english_policy_instruction_uses_wording_not_character_count(self) -> None:
        builder = AIPromptBuilder(
            locale=Locale("en"),
            preferred_categories=[],
            summary_line_target_len=120,
        )

        kwargs = builder.build_categorization_prompt_kwargs(
            shard_article_count=20,
            total_article_count=20,
            preferred_categories=[],
            suggested_categories=[],
            used_categories=[],
            example_output='{"perArticle":[{"id":1,"category":"..."}]}',
            articles_json="[]",
        )

        self.assertIn("Categories must be short and generic, like section names.", kwargs["categorization_policy_instruction"])
        self.assertIn("Usually keep labels to 1-3 words.", kwargs["categorization_policy_instruction"])
        self.assertNotIn("about 2-4 words", kwargs["categorization_policy_instruction"])

    def test_missing_prompt_raises_instead_of_falling_back(self) -> None:
        builder = AIPromptBuilder(
            locale=MissingPromptLocale(),
            preferred_categories=[],
            summary_line_target_len=120,
        )

        with self.assertRaises(KeyError):
            builder.build_summarization_request([])


if __name__ == "__main__":
    unittest.main()
