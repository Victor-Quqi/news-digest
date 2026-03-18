from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any

from .config import AIConfig
from .utils import json_dumps


class AIDebugSink:
    def __init__(
        self,
        ai_config: AIConfig,
        logger: logging.Logger,
        *,
        capture_all: bool = False,
    ) -> None:
        self._logger = logger
        self._dump_on_error = bool(ai_config.debug_dump_on_error)
        self._dump_all = bool(ai_config.debug_dump_all or capture_all)
        self._dump_dir = Path(ai_config.debug_dump_dir)
        self._dump_max_bytes = max(int(ai_config.debug_dump_max_bytes), 0)
        self._retention_days = max(int(ai_config.debug_dump_retention_days), 0)
        self._max_files = max(int(ai_config.debug_dump_max_files), 0)

        self._run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self._file_index = 1
        self._dump_file = self._file_path(self._file_index)

        self._cleanup_debug_files()
        if self._dump_on_error or self._dump_all:
            self._logger.info(
                "AI debug log file: %s (capture_all=%s, on_error=%s, max_bytes=%d, retention_days=%d, max_files=%d)",
                self._dump_file.as_posix(),
                self._dump_all,
                self._dump_on_error,
                self._dump_max_bytes,
                self._retention_days,
                self._max_files,
            )

    def dump(self, event: str, payload: dict[str, Any], *, force: bool = False) -> None:
        if not self._should_dump(force=force):
            return

        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "run_id": self._run_id,
            "event": event,
            "payload": self._to_jsonable(payload),
        }
        try:
            line = f"{json_dumps(record)}\n"
            line_bytes = len(line.encode("utf-8"))
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            self._rotate_file_if_needed(line_bytes)
            with self._dump_file.open("a", encoding="utf-8") as file:
                file.write(line)
        except Exception as exc:
            self._logger.warning("Failed to write AI debug log: %s", exc)

    def extract_file_path(self) -> str:
        return self._dump_file.as_posix()

    def _should_dump(self, *, force: bool) -> bool:
        return self._dump_all or (force and self._dump_on_error)

    def _file_path(self, index: int) -> Path:
        if self._dump_max_bytes <= 0:
            return self._dump_dir / f"{self._run_id}.jsonl"
        return self._dump_dir / f"{self._run_id}-{index:03d}.jsonl"

    def _rotate_file_if_needed(self, incoming_bytes: int) -> None:
        if self._dump_max_bytes <= 0:
            return
        try:
            current_size = self._dump_file.stat().st_size
        except FileNotFoundError:
            current_size = 0

        if current_size > 0 and current_size + max(incoming_bytes, 0) > self._dump_max_bytes:
            self._file_index += 1
            self._dump_file = self._file_path(self._file_index)
            self._logger.debug("AI debug log rotated to: %s", self._dump_file.as_posix())

    def _cleanup_debug_files(self) -> None:
        try:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            files = [path for path in self._dump_dir.glob("*.jsonl") if path.is_file()]
            removed = 0

            if self._retention_days > 0:
                cutoff = datetime.now() - timedelta(days=self._retention_days)
                for path in files:
                    try:
                        mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    except Exception:
                        continue
                    if mtime < cutoff and self._delete_file(path):
                        removed += 1
                files = [path for path in files if path.exists()]

            if self._max_files > 0 and len(files) > self._max_files:
                ordered = sorted(
                    files,
                    key=lambda path: path.stat().st_mtime if path.exists() else 0,
                    reverse=True,
                )
                for path in ordered[self._max_files :]:
                    if self._delete_file(path):
                        removed += 1

            if removed > 0:
                self._logger.info("AI debug log cleanup: removed %d old files", removed)
        except Exception as exc:
            self._logger.warning("AI debug log cleanup failed: %s", exc)

    def _delete_file(self, path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _to_jsonable(self, data: Any) -> Any:
        if data is None or isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, dict):
            return {str(key): self._to_jsonable(value) for key, value in data.items()}
        if isinstance(data, (list, tuple, set)):
            return [self._to_jsonable(item) for item in data]
        return str(data)
