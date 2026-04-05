from __future__ import annotations

import unittest

from src.ai_processor import AIProcessor


class Phase1AlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = object.__new__(AIProcessor)

    def test_low_confidence_cross_match_does_not_fail(self) -> None:
        mismatch = AIProcessor._find_phase1_alignment_mismatch(
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
        mismatch = AIProcessor._find_phase1_alignment_mismatch(
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


if __name__ == "__main__":
    unittest.main()
