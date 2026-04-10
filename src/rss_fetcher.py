from __future__ import annotations

import asyncio
import calendar
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List

import aiohttp
import feedparser

from .config import RSSSource
from .models import Article
from .utils import PipelineTimer


USER_AGENT = "news-digest/1.0 (+https://example.local)"


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_date_str(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None

    try:
        dt = parsedate_to_datetime(value)
        return _as_utc(dt)
    except Exception:
        pass

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _as_utc(dt)
    except Exception:
        return None


def _parse_entry_date(entry: feedparser.FeedParserDict) -> datetime | None:
    for parsed_key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(parsed_key)
        if parsed:
            ts = calendar.timegm(parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc)

    for str_key in ("published", "updated", "created"):
        dt = _parse_date_str(entry.get(str_key, ""))
        if dt:
            return dt

    return None


def _extract_content(entry: feedparser.FeedParserDict) -> str:
    summary = str(entry.get("summary", "") or "")
    content_list = entry.get("content", []) or []
    candidates = [summary]
    for block in content_list:
        if isinstance(block, dict):
            candidates.append(str(block.get("value", "") or ""))
        else:
            candidates.append(str(block or ""))
    candidates = [x.strip() for x in candidates if str(x or "").strip()]
    return max(candidates, key=len) if candidates else ""


async def _fetch_single_source(
    session: aiohttp.ClientSession,
    source: RSSSource,
    logger: logging.Logger,
    timer: PipelineTimer,
    missing_pub_date_strict: bool,
    timeout_seconds: int = 20,
    max_retry: int = 3,
    retry_interval_seconds: int = 5,
) -> List[Article]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {"User-Agent": USER_AGENT}

    async with timer.async_stage(f"    {source.name}"):
        for attempt in range(1, max_retry + 1):
            try:
                async with session.get(source.url, timeout=timeout, headers=headers) as resp:
                    resp.raise_for_status()
                    raw = await resp.text()

                parsed = feedparser.parse(raw)
                items: List[Article] = []
                for entry in parsed.entries:
                    title = str(entry.get("title", "") or "").strip()
                    link = str(entry.get("link", "") or "").strip()
                    content = _extract_content(entry)
                    if not title or not link:
                        continue
                    pub_date = _parse_entry_date(entry)
                    if pub_date is None:
                        if missing_pub_date_strict:
                            logger.warning(
                                "RSS entry missing publish time, dropping item: source=%s, title=%s",
                                source.name,
                                title,
                            )
                            continue
                        pub_date = datetime.now(timezone.utc)
                        logger.warning(
                            "RSS entry missing publish time, using current time: source=%s, title=%s",
                            source.name,
                            title,
                        )
                    items.append(
                        Article(
                            title=title,
                            link=link,
                            pub_date=pub_date,
                            content=content,
                            source=source.name,
                        )
                    )

                logger.info("RSS fetch success: %s, count=%d", source.name, len(items))
                return items
            except Exception as exc:
                if attempt < max_retry:
                    logger.warning(
                        "RSS fetch failed, retrying: %s (attempt %d/%d), error=%s",
                        source.name,
                        attempt,
                        max_retry,
                        repr(exc),
                    )
                    await asyncio.sleep(retry_interval_seconds)
                else:
                    logger.error("RSS fetch ultimately failed: %s, error=%s", source.name, repr(exc))
                    return []

    return []


async def fetch_all_rss(
    sources: List[RSSSource],
    logger: logging.Logger,
    *,
    missing_pub_date_strict: bool = True,
    timer: PipelineTimer | None = None,
) -> List[Article]:
    if not sources:
        return []

    if timer is None:
        timer = PipelineTimer(enabled=False)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_single_source(
                session,
                source,
                logger,
                timer,
                missing_pub_date_strict=missing_pub_date_strict,
            )
            for source in sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles: List[Article] = []
    for source, result in zip(sources, results):
        if isinstance(result, Exception):
            logger.error("RSS task exception: %s, error=%s", source.name, result)
            continue
        all_articles.extend(result)

    logger.info("RSS fetch complete: sources=%d, total_articles=%d", len(sources), len(all_articles))
    return all_articles
