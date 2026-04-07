from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.ai_processor import AIProcessor
from src.config import AIRetryTarget


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


if __name__ == "__main__":
    unittest.main()
