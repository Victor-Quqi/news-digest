from __future__ import annotations

import difflib
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List

from bs4 import BeautifulSoup

from .models import Article, CleanedArticle
from .utils import PipelineTimer, normalize_title_key, normalize_url, now_in_tz, to_timezone


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style", "noscript", "img"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _title_fingerprint(title: str) -> str:
    base = normalize_title_key(title)
    # Strip separators to reduce dedup misses from punctuation variants
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", base)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", normalize_title_key(text)))


def _title_similarity(title_a: str, title_b: str) -> float:
    a = _title_fingerprint(title_a)
    b = _title_fingerprint(title_b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    seq = difflib.SequenceMatcher(None, a, b).ratio()
    toks_a = _token_set(title_a)
    toks_b = _token_set(title_b)
    jaccard = (
        len(toks_a & toks_b) / len(toks_a | toks_b)
        if toks_a and toks_b
        else 0.0
    )
    return max(seq, jaccard)


def _content_similarity(content_a: str, content_b: str) -> float:
    a = re.sub(r"\s+", " ", (content_a or "")).strip()[:600]
    b = re.sub(r"\s+", " ", (content_b or "")).strip()[:600]
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_fuzzy_duplicate(a: Article, b: Article) -> bool:
    title_sim = _title_similarity(a.title, b.title)
    if title_sim < 0.90:
        return False
    if title_sim >= 0.97:
        return True

    # For medium-high title similarity, also check content similarity to avoid deleting "same template, different event"
    return _content_similarity(a.content, b.content) >= 0.78


def _dedupe_articles(articles: List[Article]) -> List[Article]:
    best_by_link: Dict[str, Article] = {}
    best_by_title: Dict[str, Article] = {}

    def choose_preferred(old: Article, new: Article) -> Article:
        if len(new.content or "") != len(old.content or ""):
            return new if len(new.content or "") > len(old.content or "") else old
        return new if new.pub_date >= old.pub_date else old

    for item in articles:
        link_key = normalize_url(item.link)
        title_key = normalize_title_key(item.title)

        if link_key:
            if link_key in best_by_link:
                best_by_link[link_key] = choose_preferred(best_by_link[link_key], item)
            else:
                best_by_link[link_key] = item

        if title_key:
            if title_key in best_by_title:
                best_by_title[title_key] = choose_preferred(best_by_title[title_key], item)
            else:
                best_by_title[title_key] = item

    merged = {}
    for article in best_by_link.values():
        merged[id(article)] = article
    for article in best_by_title.values():
        merged[id(article)] = article

    # Phase 2: fuzzy title dedup (cross-source same article, minor title rewrites)
    candidates = list(merged.values())
    deduped: List[Article] = []
    for item in candidates:
        merged_into_existing = False
        for idx, existed in enumerate(deduped):
            if _is_fuzzy_duplicate(item, existed):
                deduped[idx] = choose_preferred(existed, item)
                merged_into_existing = True
                break
        if not merged_into_existing:
            deduped.append(item)

    return deduped


def clean_articles(
    articles: List[Article],
    hours_back: int = 24,
    max_content_length: int = 6000,
    timezone_name: str = "Asia/Shanghai",
    logger: logging.Logger | None = None,
    *,
    timer: PipelineTimer | None = None,
) -> List[CleanedArticle]:
    if not articles:
        return []

    if timer is None:
        timer = PipelineTimer(enabled=False)

    with timer.stage("  Filter"):
        now = now_in_tz(timezone_name)
        threshold = now - timedelta(hours=hours_back)

        normalized: List[Article] = []
        for item in articles:
            if not (item.title or "").strip():
                continue
            if not (item.link or "").strip():
                continue

            cleaned_content = _html_to_text(item.content)
            if not cleaned_content:
                continue

            pub_date_local = to_timezone(item.pub_date, timezone_name)
            if pub_date_local < threshold:
                continue

            normalized.append(
                Article(
                    title=(item.title or "").strip(),
                    link=(item.link or "").strip(),
                    pub_date=pub_date_local,
                    content=cleaned_content[:max_content_length],
                    source=(item.source or "").strip(),
                )
            )

    with timer.stage("  Dedup"):
        deduped = _dedupe_articles(normalized)
        deduped.sort(key=lambda x: x.pub_date, reverse=True)

        cleaned = [
            CleanedArticle(
                id=i + 1,
                title=item.title,
                link=item.link,
                pub_date=item.pub_date,
                content=item.content,
                source=item.source,
            )
            for i, item in enumerate(deduped)
        ]

    if logger:
        logger.info(
            "Cleaning complete: input=%d, filtered=%d, deduped=%d",
            len(articles),
            len(normalized),
            len(cleaned),
        )

    return cleaned
