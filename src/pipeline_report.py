from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .i18n import Locale
from .models import ProcessedResult


@dataclass
class PipelineReport:
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        text = str(warning or "").strip()
        if text and text not in self.warnings:
            self.warnings.append(text)

    def extend_warnings(self, warnings: Sequence[str]) -> None:
        for warning in warnings:
            self.add_warning(warning)


class PipelineRunError(Exception):
    def __init__(self, cause: BaseException, warnings: Sequence[str]) -> None:
        self.cause = cause
        self.warnings = list(warnings)
        super().__init__(_exception_summary(cause))


def _fallback_text(locale: Locale, key: str, default: str) -> str:
    text = str(locale.fallback_texts.get(key, "") or "").strip()
    return text or default


def _exception_summary(exc: BaseException, max_len: int = 500) -> str:
    text = str(exc or "").strip()
    summary = f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__
    summary = " ".join(summary.split())
    if len(summary) <= max_len:
        return summary
    return summary[: max_len - 3].rstrip() + "..."


def unwrap_pipeline_error(exc: BaseException) -> tuple[BaseException, list[str]]:
    if isinstance(exc, PipelineRunError):
        return exc.cause, list(exc.warnings)
    return exc, []


def attach_report_warnings(
    result: ProcessedResult,
    warnings: Sequence[str],
    *,
    include_warnings: bool,
    locale: Locale,
) -> None:
    if not include_warnings:
        return

    seen = set(result.warnings)
    for warning in warnings:
        text = str(warning or "").strip()
        if not text or text in seen:
            continue
        result.warnings.append(text)
        seen.add(text)


def build_failure_result(
    exc: BaseException,
    *,
    warnings: Sequence[str],
    locale: Locale,
) -> ProcessedResult:
    error_summary = _exception_summary(exc)
    error_template = _fallback_text(locale, "pipeline_error", "Error: {error}")
    try:
        error_line = error_template.format(error=error_summary)
    except Exception:
        error_line = f"{error_template} {error_summary}".strip()

    merged_warnings = list(warnings)
    if error_line not in merged_warnings:
        merged_warnings.append(error_line)

    return ProcessedResult(
        articles=[],
        categories=[],
        summary_lines=[
            _fallback_text(
                locale,
                "pipeline_failed",
                "News digest generation failed before a normal email could be produced.",
            )
        ],
        degraded=False,
        warnings=merged_warnings,
        subject_label=locale.t("Run Failed"),
    )
