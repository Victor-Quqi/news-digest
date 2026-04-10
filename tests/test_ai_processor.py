from __future__ import annotations

import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace

from src.ai_processor import AIProcessor
from src.config import AIRetryTarget
from src.i18n import Locale
from src.models import CleanedArticle


class SummarizationAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = object.__new__(AIProcessor)

    def test_low_confidence_cross_match_does_not_fail(self) -> None:
        mismatch = AIProcessor._find_summarization_alignment_mismatch(
            self.processor,
            article_id=9,
            one_line="特朗普称美国可在两到三周内结束对伊朗的军事打击。",
            title_by_id={
                1: "特朗普称美国将结束对伊朗行动。",
                9: "多家银行下调短期存款利率，流动性预期偏松。",
            },
        )

        self.assertIsNone(mismatch)

    def test_high_confidence_cross_match_still_fails(self) -> None:
        mismatch = AIProcessor._find_summarization_alignment_mismatch(
            self.processor,
            article_id=9,
            one_line="特朗普称美国可在两到三周内结束对伊朗的军事打击。",
            title_by_id={
                1: "特朗普称美国将在两到三周内结束伊朗行动。",
                9: "多家银行下调短期存款利率，流动性预期偏松。",
            },
        )

        self.assertEqual(mismatch, (1, 0.0, mismatch[2]))
        self.assertGreaterEqual(mismatch[2], 0.45)


class RetryTargetSwitchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = object.__new__(AIProcessor)
        self.processor.cfg = SimpleNamespace(retry_target_failure_threshold=3)

    def test_switch_after_three_failures(self) -> None:
        targets = [
            AIRetryTarget(name="primary", base_url="https://a.example/v1", api_key="k1", model="m1"),
            AIRetryTarget(name="backup", base_url="https://b.example/v1", api_key="k2", model="m2"),
        ]

        candidate_index, candidate_failures, switched = AIProcessor._advance_retry_target_state(
            self.processor,
            targets=targets,
            candidate_index=0,
            candidate_failures=2,
        )

        self.assertEqual(candidate_index, 1)
        self.assertEqual(candidate_failures, 0)
        self.assertTrue(switched)

    def test_last_target_stays_active_after_threshold(self) -> None:
        targets = [
            AIRetryTarget(name="primary", base_url="https://a.example/v1", api_key="k1", model="m1"),
            AIRetryTarget(name="backup", base_url="https://b.example/v1", api_key="k2", model="m2"),
        ]

        candidate_index, candidate_failures, switched = AIProcessor._advance_retry_target_state(
            self.processor,
            targets=targets,
            candidate_index=1,
            candidate_failures=2,
        )

        self.assertEqual(candidate_index, 1)
        self.assertEqual(candidate_failures, 3)
        self.assertFalse(switched)


class RetryEligibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = object.__new__(AIProcessor)
        self.processor.cfg = SimpleNamespace(
            retry_target_failure_threshold=3,
            retry_error_keywords=[],
        )

    def test_non_transient_error_is_not_retried_by_default(self) -> None:
        self.assertFalse(
            AIProcessor._should_retry_call_error(
                self.processor,
                ValueError("model returned empty content"),
            )
        )

    def test_configured_keyword_allows_retry(self) -> None:
        self.processor.cfg.retry_error_keywords = ["empty content"]
        self.assertTrue(
            AIProcessor._should_retry_call_error(
                self.processor,
                ValueError("model returned empty content"),
            )
        )


class OverviewValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = object.__new__(AIProcessor)
        self.processor.cfg = SimpleNamespace(overview_local_fallback=False)
        self.processor._summary_line_target_len = 120
        self.processor._summary_line_hard_limit = 140.0
        self.processor._summary_line_soft_limit = 168.0

    def test_placeholder_like_first_line_is_no_longer_rejected(self) -> None:
        lines = AIProcessor._validate_overview(
            self.processor,
            ["总述", "细节一[1]", "细节二[2]", "细节三[1]", "细节四[2]"],
            {1, 2},
            locale=SimpleNamespace(overview_placeholders={"总述"}),
        )

        self.assertEqual(len(lines), 5)

    def test_four_line_overview_is_allowed(self) -> None:
        lines = AIProcessor._validate_overview(
            self.processor,
            ["总述", "细节一[1]", "细节二[2]", "细节三[1]"],
            {1, 2},
            locale=SimpleNamespace(overview_placeholders=set()),
        )

        self.assertEqual(len(lines), 4)

    def test_first_line_with_refs_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "first line must not contain references"):
            AIProcessor._validate_overview(
                self.processor,
                ["总述[1]", "细节一[1]", "细节二[2]", "细节三[1]"],
                {1, 2},
                locale=SimpleNamespace(overview_placeholders=set()),
            )

    def test_overview_headline_rejects_refs(self) -> None:
        with self.assertRaisesRegex(ValueError, "first line must not contain references"):
            AIProcessor._validate_overview_headline(
                self.processor,
                {"headline": "总述[1]"},
                locale=SimpleNamespace(lang="zh"),
            )

    def test_overview_headline_rejects_json_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "plain text, not JSON"):
            AIProcessor._validate_overview_headline(
                self.processor,
                '{"headline":"总述"}',
                locale=SimpleNamespace(lang="zh"),
            )

    def test_overview_groups_require_trailing_refs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must end with references"):
            AIProcessor._validate_overview_groups(
                self.processor,
                ["细节一[1]", "细节二[2]尾巴", "细节三[1][2]"],
                {1, 2},
                locale=SimpleNamespace(lang="zh"),
            )

    def test_postprocess_no_longer_uses_local_fallback_for_short_output(self) -> None:
        self.processor.cfg = SimpleNamespace(overview_local_fallback=True)

        with self.assertRaisesRegex(ValueError, "too few lines"):
            AIProcessor._postprocess_overview_lines(
                self.processor,
                ["总述", "细节一[1][2]", "细节二[1][2]"],
                [],
                locale=SimpleNamespace(lang="zh"),
            )


class CategorizationPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = object.__new__(AIProcessor)
        self.processor.cfg = SimpleNamespace(preferred_categories=[])
        self.processor.locale = Locale("zh")

    def test_dynamic_category_cap_matches_examples(self) -> None:
        self.assertEqual(AIProcessor._categorization_max_categories(self.processor, 20), 6)
        self.assertEqual(AIProcessor._categorization_max_categories(self.processor, 40), 8)
        self.assertEqual(AIProcessor._categorization_max_categories(self.processor, 100), 10)

    def test_first_shard_prompt_omits_empty_optional_blocks(self) -> None:
        captured: dict[str, str] = {}

        class DummyTransport:
            async def request_json(
                self,
                system_prompt: str,
                user_prompt: str,
                schema: dict,
                step: str,
                *,
                target: AIRetryTarget,
            ) -> dict:
                captured["system_prompt"] = system_prompt
                captured["user_prompt"] = user_prompt
                return {"perArticle": [{"id": 1, "category": "宏观政策"}]}

        self.processor._transport = DummyTransport()
        article = CleanedArticle(
            id=1,
            title="美国就业数据走强",
            link="https://example.com/1",
            pub_date=datetime(2026, 4, 10, 8, 0, 0),
            source="Reuters",
            content="就业与通胀数据共同影响美联储判断。",
        )

        asyncio.run(
            AIProcessor._request_categorization(
                self.processor,
                [article],
                self.processor.locale,
                AIRetryTarget(name="primary", base_url="https://example.com/v1", api_key="k", model="m"),
                category_candidates=[],
                used_categories=[],
                total_article_count=1,
            )
        )

        self.assertNotIn("preferred_categories=", captured["user_prompt"])
        self.assertNotIn("suggested_categories=", captured["user_prompt"])
        self.assertNotIn("already_used_categories=", captured["user_prompt"])
        self.assertNotIn("如果提供了 preferred_categories", captured["system_prompt"])
        self.assertNotIn("already_used_categories 是前面分片已经用过的分类名", captured["system_prompt"])

    def test_later_shard_prompt_includes_configured_and_used_categories(self) -> None:
        self.processor.cfg = SimpleNamespace(preferred_categories=["宏观政策", "科技AI"])
        prompt_kwargs = AIProcessor._build_categorization_prompt_kwargs(
            self.processor,
            locale=self.processor.locale,
            shard_article_count=20,
            total_article_count=20,
            preferred_categories=["宏观政策", "科技AI"],
            suggested_categories=[],
            used_categories=["宏观政策"],
            example_output='{"perArticle":[{"id":1,"category":"..."}]}',
            articles_json="[]",
        )

        self.assertIn("preferred_categories=", prompt_kwargs["category_candidates_block"])
        self.assertIn("already_used_categories=", prompt_kwargs["already_used_categories_block"])
        self.assertEqual(prompt_kwargs["max_categories"], 6)
        self.assertIn("优先直接使用 preferred_categories 里的分类名", prompt_kwargs["categorization_policy_instruction"])
        self.assertIn("最好 2~4 字", prompt_kwargs["categorization_policy_instruction"])

    def test_prompt_can_use_suggested_categories_without_user_config(self) -> None:
        prompt_kwargs = AIProcessor._build_categorization_prompt_kwargs(
            self.processor,
            locale=self.processor.locale,
            shard_article_count=20,
            total_article_count=20,
            preferred_categories=[],
            suggested_categories=["国际", "财经", "科技"],
            used_categories=["国际"],
            example_output='{"perArticle":[{"id":1,"category":"..."}]}',
            articles_json="[]",
        )

        self.assertIn("suggested_categories=", prompt_kwargs["category_candidates_block"])
        self.assertIn("优先直接使用 suggested_categories 里的分类名", prompt_kwargs["categorization_policy_instruction"])

    def test_english_policy_instruction_uses_natural_length_wording(self) -> None:
        prompt_kwargs = AIProcessor._build_categorization_prompt_kwargs(
            self.processor,
            locale=SimpleNamespace(lang="en"),
            shard_article_count=20,
            total_article_count=20,
            preferred_categories=[],
            suggested_categories=[],
            used_categories=[],
            example_output='{"perArticle":[{"id":1,"category":"..."}]}',
            articles_json="[]",
        )

        self.assertIn("Categories must be short and generic, like section names.", prompt_kwargs["categorization_policy_instruction"])
        self.assertIn("Usually keep labels to 1-3 words.", prompt_kwargs["categorization_policy_instruction"])
        self.assertNotIn("about 2-4 words", prompt_kwargs["categorization_policy_instruction"])


class CategorizationValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = object.__new__(AIProcessor)

    def test_zh_locale_allows_non_empty_category(self) -> None:
        rows = AIProcessor._validate_categorization(
            self.processor,
            {"perArticle": [{"id": 1, "category": "IPO"}]},
            {1},
            locale=SimpleNamespace(lang="zh"),
        )

        self.assertEqual(rows, [{"id": 1, "category": "IPO"}])

    def test_category_suggestion_requires_non_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            AIProcessor._validate_category_suggestion(
                self.processor,
                {"categories": []},
                locale=SimpleNamespace(lang="zh"),
                max_categories=6,
            )


if __name__ == "__main__":
    unittest.main()
