from __future__ import annotations

import unittest

import feedparser

from src.rss_fetcher import _parse_entry_date


class RSSFetcherDateParsingTest(unittest.TestCase):
    def test_missing_publish_time_returns_none(self) -> None:
        entry = feedparser.FeedParserDict({})

        self.assertIsNone(_parse_entry_date(entry))


if __name__ == "__main__":
    unittest.main()
