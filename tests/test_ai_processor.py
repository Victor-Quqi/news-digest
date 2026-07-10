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


class ShardPlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = object.__new__(AIProcessor)
        self.processor.cfg = SimpleNamespace(shard_max_articles=2)

    def test_article_limit_triggers_sharding_below_token_threshold(self) -> None:
        articles = [SimpleNamespace(id=index) for index in range(3)]

        should_shard = AIProcessor._should_shard(
            self.processor,
            articles,
            estimated_tokens=50,
            threshold=100,
        )

        self.assertTrue(should_shard)

    def test_shards_respect_article_limit(self) -> None:
        articles = [SimpleNamespace(id=index, token_cost=1) for index in range(5)]

        shards = AIProcessor._build_token_bounded_shards(
            self.processor,
            articles,
            threshold=100,
            estimate_tokens=lambda items: sum(item.token_cost for item in items),
        )

        self.assertEqual([len(shard) for shard in shards], [2, 2, 1])

    def test_shards_respect_token_budget(self) -> None:
        self.processor.cfg.shard_max_articles = 10
        articles = [SimpleNamespace(id=index, token_cost=6) for index in range(3)]

        shards = AIProcessor._build_token_bounded_shards(
            self.processor,
            articles,
            threshold=10,
            estimate_tokens=lambda items: sum(item.token_cost for item in items),
        )

        self.assertEqual([len(shard) for shard in shards], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
