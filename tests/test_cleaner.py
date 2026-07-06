from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

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


class CleanerNearDuplicateTest(unittest.TestCase):
    def article(
        self,
        title: str,
        link: str,
        *,
        minutes_ago: int = 0,
        content: str = "正文内容足够用于去重判断。",
    ) -> Article:
        return Article(
            title=title,
            link=link,
            pub_date=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
            content=f"<p>{content}</p>",
            source="Test Feed",
        )

    def test_dedupes_same_spacex_fact_with_rewritten_title(self) -> None:
        articles = [
            self.article(
                "马斯克否认SpaceX在IPO前向投资者展示AI手持设备原型的报道",
                "https://example.com/a",
                minutes_ago=10,
            ),
            self.article(
                "马斯克称SpaceX在IPO前展示AI手持设备原型的报道完全虚假",
                "https://example.com/b",
            ),
        ]

        cleaned = clean_articles(articles)

        self.assertEqual(len(cleaned), 1)
        self.assertIn("SpaceX", cleaned[0].title)

    def test_dedupes_same_gold_forecast_with_extra_context(self) -> None:
        articles = [
            self.article(
                "摩根大通将黄金2026年第四季度均价预期定为4500美元/盎司",
                "https://example.com/a",
                minutes_ago=10,
            ),
            self.article(
                "摩根大通预计黄金短期震荡后回升，2026年第四季度均价达4500美元/盎司",
                "https://example.com/b",
            ),
        ]

        cleaned = clean_articles(articles)

        self.assertEqual(len(cleaned), 1)
        self.assertIn("摩根大通", cleaned[0].title)

    def test_keeps_same_company_different_events(self) -> None:
        articles = [
            self.article("特斯拉第二季度交付量超市场预期", "https://example.com/a"),
            self.article("特斯拉因自动驾驶事故遭监管调查", "https://example.com/b"),
        ]

        cleaned = clean_articles(articles)

        self.assertEqual(len(cleaned), 2)

    def test_keeps_similar_titles_with_conflicting_numbers(self) -> None:
        articles = [
            self.article(
                "摩根大通预计黄金2026年第四季度均价达4500美元/盎司",
                "https://example.com/a",
            ),
            self.article(
                "摩根大通预计黄金2026年第四季度均价达4300美元/盎司",
                "https://example.com/b",
            ),
        ]

        cleaned = clean_articles(articles)

        self.assertEqual(len(cleaned), 2)


if __name__ == "__main__":
    unittest.main()
