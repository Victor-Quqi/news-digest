from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.ai_transport import AITransport


class StructuredPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = object.__new__(AITransport)
        self.transport._cfg = SimpleNamespace(
            structured_output_summarization_formats=[],
            structured_output_summarization_policy="strict",
            structured_output_overview_formats=["json_object"],
            structured_output_overview_policy="prefer",
        )
        self.transport._logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

    def test_overview_variants_share_overview_policy(self) -> None:
        for step in ("overview", "overview_headline", "overview_groups"):
            with self.subTest(step=step):
                formats, policy, fallback_to_text = AITransport._structured_policy(
                    self.transport,
                    step,
                )
                self.assertEqual(formats, ["json_object"])
                self.assertEqual(policy, "prefer")
                self.assertTrue(fallback_to_text)

    def test_category_suggestion_shares_summarization_policy(self) -> None:
        formats, policy, fallback_to_text = AITransport._structured_policy(
            self.transport,
            "category_suggestion",
        )

        self.assertEqual(formats, [])
        self.assertEqual(policy, "strict")
        self.assertFalse(fallback_to_text)


if __name__ == "__main__":
    unittest.main()
