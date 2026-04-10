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


class SummarizationAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = make_outputs()

    def test_low_confidence_cross_match_does_not_fail(self) -> None:
        mismatch = self.outputs.find_summarization_alignment_mismatch(
            article_id=9,
            one_line="特朗普称美国可在两到三周内结束对伊朗的军事打击。",
            title_by_id={
                1: "特朗普称美国将结束对伊朗行动。",
                9: "多家银行下调短期存款利率，流动性预期偏松。",
            },
        )

        self.assertIsNone(mismatch)

    def test_high_confidence_cross_match_still_fails(self) -> None:
        mismatch = self.outputs.find_summarization_alignment_mismatch(
            article_id=9,
            one_line="特朗普称美国可在两到三周内结束对伊朗的军事打击。",
            title_by_id={
                1: "特朗普称美国将在两到三周内结束伊朗行动。",
                9: "多家银行下调短期存款利率，流动性预期偏松。",
            },
        )

        self.assertEqual(mismatch, (1, 0.0, mismatch[2]))
        self.assertGreaterEqual(mismatch[2], 0.45)


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
