from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.ai_processor import AIProcessor
from src.config import AIRetryTarget


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
        self.processor.cfg = SimpleNamespace(retry_error_keywords=[])

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
