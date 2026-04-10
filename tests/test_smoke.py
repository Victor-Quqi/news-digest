from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.email_sender import render_email_html
from src.i18n import Locale
from src.models import ProcessedArticle, ProcessedResult


class ExampleConfigSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_MODEL"] = "gpt-5.4"
        os.environ.pop("OPENAI_BACKUP_API_KEY", None)
        os.environ.pop("OPENAI_BACKUP_BASE_URL", None)
        os.environ.pop("OPENAI_BACKUP_MODEL", None)
        os.environ["SMTP_HOST"] = "smtp.example.com"
        os.environ["SMTP_PORT"] = "587"
        os.environ["SMTP_USER"] = "tester@example.com"
        os.environ["SMTP_PASSWORD"] = "secret"
        os.environ["EMAIL_TO"] = "recipient@example.com"
        os.environ["RSSHUB_BASE_URL"] = "http://localhost:1200"
        os.environ["NEWS_DIGEST_LOCALE"] = "zh"

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
                    category="",
                )
            ],
            categories=[],
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
        self.assertEqual(len(cfg.ai.summarization_retry_targets), 1)
        self.assertEqual(cfg.ai.summarization_retry_targets[0].name, "primary")
        self.assertEqual(len(cfg.ai.overview_retry_targets), 1)
        self.assertEqual(cfg.ai.overview_retry_targets[0].model, cfg.env.openai_model)
        self.assertEqual(cfg.ai.preferred_categories, [])
        self.assertTrue(cfg.ai.categorization_strict)
        self.assertTrue(cfg.filter.rss_missing_pub_date_strict)
        self.assertIn("Example overview", html)
        self.assertIn("https://example.com/article", html)
        self.assertNotIn("其他", html)

    def test_env_locale_and_optional_lists(self) -> None:
        config_path = Path("tests") / "_tmp_config.yaml"
        sources_path = Path("tests") / "_tmp_sources.yaml"
        try:
            config_path.write_text(
                "\n".join(
                    [
                        "locale: ${NEWS_DIGEST_LOCALE}",
                        "email:",
                        '  to: ["${EMAIL_TO}"]',
                        "ai:",
                        "  structured_output_summarization_formats: []",
                        "  structured_output_overview_formats: []",
                        "  preferred_categories: []",
                        "  categorization_strict: false",
                        "filter:",
                        "  rss_missing_pub_date_strict: false",
                    ]
                ),
                encoding="utf-8",
            )
            sources_path.write_text(
                "\n".join(
                    [
                        "- bad",
                        "- name: Missing URL",
                        "- url: https://example.com/feed.xml",
                        "  name: Example Feed",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_config(str(config_path.resolve()), sources_path.name)
        finally:
            config_path.unlink(missing_ok=True)
            sources_path.unlink(missing_ok=True)

        self.assertEqual(cfg.locale, "zh")
        self.assertEqual(cfg.ai.structured_output_summarization_formats, [])
        self.assertEqual(cfg.ai.structured_output_overview_formats, [])
        self.assertEqual(cfg.ai.preferred_categories, [])
        self.assertFalse(cfg.ai.categorization_strict)
        self.assertFalse(cfg.filter.rss_missing_pub_date_strict)
        self.assertEqual(len(cfg.rss_sources), 1)


if __name__ == "__main__":
    unittest.main()
