from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytz


TRACKING_QUERY_PREFIXES = (
    "utm_",
    "spm",
    "from",
    "feature",
    "ref",
    "source",
)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def setup_logging(
    level: str = "INFO",
    file_path: str = "logs/news-digest.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    ensure_parent_dir(file_path)

    logger = logging.getLogger("news_digest")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )

    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_timezone(name: str) -> pytz.BaseTzInfo:
    return pytz.timezone(name)


def now_in_tz(tz_name: str) -> datetime:
    tz = get_timezone(tz_name)
    return datetime.now(tz)


def to_timezone(dt: datetime, tz_name: str) -> datetime:
    tz = get_timezone(tz_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz)


def format_pub_datetime(dt: datetime, tz_name: str) -> str:
    local_dt = to_timezone(dt, tz_name)
    return local_dt.strftime("%Y-%m-%d %H:%M")


def today_str(tz_name: str) -> str:
    return now_in_tz(tz_name).strftime("%Y-%m-%d")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_title_key(title: str) -> str:
    return normalize_whitespace((title or "").lower())


def normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""

    split = urlsplit(raw)
    query_items = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        key_lower = key.lower()
        if key_lower.startswith(TRACKING_QUERY_PREFIXES):
            continue
        query_items.append((key, value))

    cleaned_query = urlencode(query_items, doseq=True)
    normalized = urlunsplit(
        (
            split.scheme.lower(),
            split.netloc.lower(),
            split.path.rstrip("/"),
            cleaned_query,
            "",  # drop fragment
        )
    )
    return normalized


def json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


@contextmanager
def file_lock(lock_path: str):
    ensure_parent_dir(lock_path)
    handle = None
    try:
        handle = open(lock_path, "a+", encoding="utf-8")

        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            handle.write("0")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise FileExistsError(lock_path) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise FileExistsError(lock_path) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()
class PipelineTimer:
    """Records elapsed time for named pipeline stages.

    Stage names use leading spaces to indicate nesting depth:
      "RSS fetch"         — top level (depth 0)
      "  Phase 1"         — depth 1 (child of preceding depth-0)
      "    Shard 1"       — depth 2 (child of preceding depth-1)
      "      API"         — depth 3 (child of preceding depth-2)

    Records arrive in completion order (deepest-first because inner
    context managers exit before outer ones).  The summary method
    rebuilds the tree and prints parent-then-children order.
    """

    INDENT = "  "  # 2 spaces per depth level

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._records: list[tuple[str, float]] = []

    def _depth(self, name: str) -> int:
        stripped = name.lstrip(" ")
        return (len(name) - len(stripped)) // len(self.INDENT)

    def record(self, name: str, duration: float) -> None:
        """Manually add a timing record (for accumulated measurements)."""
        if self.enabled:
            self._records.append((name, duration))

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._records.append((name, time.monotonic() - t0))

    @asynccontextmanager
    async def async_stage(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._records.append((name, time.monotonic() - t0))

    def _reorder(self) -> list[tuple[str, float]]:
        """Rebuild parent → children order from completion-order records.

        Uses a stack: when a record at depth D arrives, all stack items
        at depth > D are popped and attached as its children.
        """
        Node = tuple[str, float, list]  # (name, duration, children)
        stack: list[Node] = []

        for name, dur in self._records:
            depth = self._depth(name)
            children: list[Node] = []
            while stack and self._depth(stack[-1][0]) > depth:
                children.append(stack.pop())
            children.reverse()
            stack.append((name, dur, children))

        def flatten(nodes: list[Node]) -> list[tuple[str, float]]:
            result: list[tuple[str, float]] = []
            for n, d, ch in nodes:
                result.append((n, d))
                result.extend(flatten(ch))
            return result

        return flatten(stack)

    def summary(self, logger: logging.Logger) -> None:
        if not self.enabled or not self._records:
            return
        ordered = self._reorder()
        total = sum(d for n, d in ordered if self._depth(n) == 0)
        label_width = max(len(n) for n, _ in ordered)
        lines = ["Pipeline timing:"]
        for name, duration in ordered:
            lines.append(f"  {name:<{label_width}} : {duration:>7.2f}s")
        lines.append(f"  {'Total':<{label_width}} : {total:>7.2f}s")
        logger.info("\n".join(lines))
