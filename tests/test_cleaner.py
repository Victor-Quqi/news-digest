from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.cleaner import _truncate_content_keep_head_tail, clean_articles
from src.models import Article


class CleanerContentTruncationTest(unittest.TestCase):
    def test_head_tail_truncation_keeps_both_ends(self) -> None:
        text = "HEAD" + ("x" * 20) + "TAIL"

        truncated = _truncate_content_keep_head_tail(text, 12)

        self.assertTrue(truncated.startswith("HEAD"))
        self.assertTrue(truncated.endswith("TAIL"))
        self.assertIn("…", truncated)

    def test_clean_articles_uses_head_tail_truncation(self) -> None:
        article = Article(
            title="Example",
            link="https://example.com/article",
            pub_date=datetime.now(timezone.utc),
            content="<p>HEAD" + ("x" * 40) + "TAIL</p>",
            source="Example Feed",
        )

        cleaned = clean_articles([article], max_content_length=12)

        self.assertEqual(len(cleaned), 1)
        self.assertTrue(cleaned[0].content.startswith("HEAD"))
        self.assertTrue(cleaned[0].content.endswith("TAIL"))


if __name__ == "__main__":
    unittest.main()
