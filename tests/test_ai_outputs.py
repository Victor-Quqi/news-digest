from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from src.ai_outputs import AIOutputProcessor
from src.i18n import Locale


def make_outputs(locale: Locale | SimpleNamespace | None = None) -> AIOutputProcessor:
    return AIOutputProcessor(
        logger=logging.getLogger("test.ai_outputs"),
        locale=locale or Locale("zh"),
        preferred_categories=[],
        one_line_hard_units=42.0,
        one_line_soft_units=50.0,
        one_line_trim_target_units=48.0,
        summary_line_target_len=120,
        summary_line_hard_limit=140.0,
        summary_line_soft_limit=168.0,
    )


class SummarizationValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = make_outputs()

    def test_accepts_structurally_valid_summaries(self) -> None:
        rows = self.outputs.validate_summarization(
            {
                "perArticle": [
                    {"id": 3, "oneLine": "英伟达正洽谈为 OpenAI 数据中心提供融资担保。"},
                    {"id": 6, "oneLine": "英伟达正讨论为 OpenAI 俄亥俄州数据中心提供财务担保。"},
                ]
            },
            expected_ids={3, 6},
        )

        self.assertEqual([row["id"] for row in rows], [3, 6])

    def test_rejects_incomplete_id_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "id set mismatch"):
            self.outputs.validate_summarization(
                {"perArticle": [{"id": 3, "oneLine": "摘要"}]},
                expected_ids={3, 6},
            )


class OverviewValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = make_outputs(SimpleNamespace(lang="zh", theme_keywords={}, fallback_texts={}))

    def test_placeholder_like_first_line_is_allowed(self) -> None:
        lines = self.outputs.validate_overview(
            ["总述", "细节一[1]", "细节二[2]", "细节三[1]", "细节四[2]"],
            {1, 2},
        )

        self.assertEqual(len(lines), 5)

    def test_four_line_overview_is_allowed(self) -> None:
        lines = self.outputs.validate_overview(
            ["总述", "细节一[1]", "细节二[2]", "细节三[1]"],
            {1, 2},
        )

        self.assertEqual(len(lines), 4)

    def test_overview_headline_rejects_refs(self) -> None:
        with self.assertRaisesRegex(ValueError, "first line must not contain references"):
            self.outputs.validate_overview_headline({"headline": "总述[1]"})

    def test_overview_headline_rejects_json_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "plain text, not JSON"):
            self.outputs.validate_overview_headline('{"headline":"总述"}')

    def test_overview_groups_require_trailing_refs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must end with references"):
            self.outputs.validate_overview_groups(
                ["细节一[1]", "细节二[2]尾巴", "细节三[1][2]"],
                {1, 2},
            )

    def test_postprocess_rejects_too_few_lines(self) -> None:
        with self.assertRaisesRegex(ValueError, "too few lines"):
            self.outputs.postprocess_overview_lines(
                ["总述", "细节一[1][2]", "细节二[1][2]"],
                [],
            )


class CategorizationValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = make_outputs()

    def test_allows_non_empty_category(self) -> None:
        rows = self.outputs.validate_categorization(
            {"perArticle": [{"id": 1, "category": "IPO"}]},
            expected_ids={1},
        )

        self.assertEqual(rows, [{"id": 1, "category": "IPO"}])

    def test_category_suggestion_requires_non_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.outputs.validate_category_suggestion(
                {"categories": []},
                max_categories=6,
            )


if __name__ == "__main__":
    unittest.main()
