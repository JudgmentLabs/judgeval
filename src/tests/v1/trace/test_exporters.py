"""Tests for JudgmentSpanExporter and NoOpJudgmentSpanExporter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace.export import SpanExportResult

from judgeval.v1.trace.exporters.judgment_span_exporter import JudgmentSpanExporter
from judgeval.v1.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)


class TestNoOpJudgmentSpanExporter:
    def test_export_returns_success(self):
        exporter = NoOpJudgmentSpanExporter()
        assert exporter.export([]) == SpanExportResult.SUCCESS

    def test_shutdown_does_not_raise(self):
        exporter = NoOpJudgmentSpanExporter()
        exporter.shutdown()

    def test_force_flush_returns_true(self):
        exporter = NoOpJudgmentSpanExporter()
        assert exporter.force_flush() is True


class TestJudgmentSpanExporter:
    def test_creates_delegate_with_correct_headers(self):
        with patch(
            "judgeval.v1.trace.exporters.judgment_span_exporter.OTLPSpanExporter"
        ) as MockOTLP:
            JudgmentSpanExporter(
                endpoint="http://test/otel/v1/traces",
                api_key="my-key",
                organization_id="my-org",
                project_id="my-proj",
            )

            MockOTLP.assert_called_once_with(
                endpoint="http://test/otel/v1/traces",
                headers={
                    "Authorization": "Bearer my-key",
                    "X-Organization-Id": "my-org",
                    "X-Project-Id": "my-proj",
                },
            )

    def test_export_delegates(self):
        with patch(
            "judgeval.v1.trace.exporters.judgment_span_exporter.OTLPSpanExporter"
        ) as MockOTLP:
            mock_delegate = MagicMock()
            mock_delegate.export.return_value = SpanExportResult.SUCCESS
            MockOTLP.return_value = mock_delegate

            exporter = JudgmentSpanExporter(
                endpoint="http://test",
                api_key="k",
                organization_id="o",
                project_id="p",
            )
            spans = [MagicMock()]
            result = exporter.export(spans)

            assert result == SpanExportResult.SUCCESS
            mock_delegate.export.assert_called_once_with(spans)

    def test_shutdown_delegates(self):
        with patch(
            "judgeval.v1.trace.exporters.judgment_span_exporter.OTLPSpanExporter"
        ) as MockOTLP:
            mock_delegate = MagicMock()
            MockOTLP.return_value = mock_delegate

            exporter = JudgmentSpanExporter(
                endpoint="http://test",
                api_key="k",
                organization_id="o",
                project_id="p",
            )
            exporter.shutdown()
            mock_delegate.shutdown.assert_called_once()
