from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone

from src.config import load_config
from src.email_sender import render_email_html
from src.i18n import Locale
from src.models import ProcessedArticle, ProcessedResult


class ExampleConfigSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["SMTP_HOST"] = "smtp.example.com"
        os.environ["SMTP_PORT"] = "587"
        os.environ["SMTP_USER"] = "tester@example.com"
        os.environ["SMTP_PASSWORD"] = "secret"
        os.environ["EMAIL_TO"] = "recipient@example.com"
        os.environ["RSSHUB_BASE_URL"] = "http://localhost:1200"

    def test_example_config_and_template_render(self) -> None:
        cfg = load_config("config.yaml.example", "sources.yaml.example")
        locale = Locale(cfg.locale)
        result = ProcessedResult(
            articles=[
                ProcessedArticle(
                    id=1,
                    title="Example title",
                    link="https://example.com/article",
                    pub_date=datetime.now(timezone.utc),
                    source="Example Feed",
                    one_line="Example summary",
                    category=locale.default_category or "Other",
                )
            ],
            categories=[locale.default_category or "Other"],
            summary_lines=["Example overview[1]"],
        )

        html = render_email_html(
            result=result,
            date_text="2026-03-19",
            locale=locale,
        )

        self.assertEqual(cfg.locale, "zh")
        self.assertEqual(len(cfg.rss_sources), 2)
        self.assertEqual(cfg.ai.transient_retry_max, 8)
        self.assertEqual(cfg.ai.schema_retry_max, 8)
        self.assertEqual(cfg.ai.retry_error_keywords, [])
        self.assertEqual(cfg.ai.retry_target_failure_threshold, 3)
        self.assertEqual(len(cfg.ai.phase1_retry_targets), 1)
        self.assertEqual(cfg.ai.phase1_retry_targets[0].name, "primary")
        self.assertEqual(len(cfg.ai.phase2_retry_targets), 1)
        self.assertEqual(cfg.ai.phase2_retry_targets[0].model, cfg.env.openai_model)
        self.assertIn("Example overview", html)
        self.assertIn("https://example.com/article", html)


if __name__ == "__main__":
    unittest.main()
