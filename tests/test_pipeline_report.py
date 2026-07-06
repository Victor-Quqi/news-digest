from __future__ import annotations

import unittest

from src.i18n import Locale
from src.pipeline_report import (
    PipelineReport,
    attach_report_warnings,
    build_failure_result,
)
from src.models import ProcessedResult


class PipelineReportTest(unittest.TestCase):
    def test_attach_warnings_keeps_normal_digest_subject_unlabeled(self) -> None:
        locale = Locale("zh")
        report = PipelineReport()
        report.add_warning("RSS source failed: Feed A")
        report.add_warning("RSS source failed: Feed A")
        result = ProcessedResult(articles=[], categories=[], summary_lines=["ok"])

        attach_report_warnings(
            result,
            report.warnings,
            include_warnings=True,
            locale=locale,
        )

        self.assertEqual(result.warnings, ["RSS source failed: Feed A"])
        self.assertEqual(result.subject_label, "")

    def test_failure_result_contains_error_warning_and_subject_label(self) -> None:
        result = build_failure_result(
            RuntimeError("boom"),
            warnings=["RSS source failed: Feed A"],
            locale=Locale("zh"),
        )

        self.assertEqual(result.subject_label, "运行失败")
        self.assertIn("本次日报生成失败", result.summary_lines[0])
        self.assertEqual(result.warnings[0], "RSS source failed: Feed A")
        self.assertIn("RuntimeError: boom", result.warnings[1])


if __name__ == "__main__":
    unittest.main()
