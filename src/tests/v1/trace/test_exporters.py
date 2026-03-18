from __future__ import annotations

from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace.export import SpanExportResult

from judgeval.v1.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)
from judgeval.v1.trace.exporters.judgment_span_exporter import JudgmentSpanExporter


class TestNoOpJudgmentSpanExporter:
    def test_export_returns_success(self):
        exp = NoOpJudgmentSpanExporter()
        result = exp.export([])
        assert result == SpanExportResult.SUCCESS

    def test_export_with_spans_returns_success(self):
        exp = NoOpJudgmentSpanExporter()
        result = exp.export([MagicMock(), MagicMock()])
        assert result == SpanExportResult.SUCCESS

    def test_force_flush_returns_true(self):
        assert NoOpJudgmentSpanExporter().force_flush() is True

    def test_shutdown_does_not_raise(self):
        NoOpJudgmentSpanExporter().shutdown()


class TestJudgmentSpanExporter:
    def _make_exporter(self):
        return JudgmentSpanExporter(
            endpoint="http://api/otel/v1/traces",
            api_key="key",
            organization_id="org",
            project_id="proj",
        )

    def test_delegate_endpoint(self):
        exp = self._make_exporter()
        assert exp._delegate._endpoint == "http://api/otel/v1/traces"

    def test_delegate_has_auth_header(self):
        exp = self._make_exporter()
        headers = {
            k.lower() if isinstance(k, str) else k: v
            for k, v in exp._delegate._headers.items()
        }
        auth = headers.get("authorization", headers.get(b"authorization", ""))
        assert "Bearer key" in auth

    def test_export_delegates(self):
        exp = self._make_exporter()
        with patch.object(
            exp._delegate, "export", return_value=SpanExportResult.SUCCESS
        ) as mock_export:
            result = exp.export([])
        mock_export.assert_called_once_with([])
        assert result == SpanExportResult.SUCCESS

    def test_shutdown_delegates(self):
        exp = self._make_exporter()
        with patch.object(exp._delegate, "shutdown") as mock_shutdown:
            exp.shutdown()
        mock_shutdown.assert_called_once()

    def test_force_flush_delegates(self):
        exp = self._make_exporter()
        with patch.object(exp._delegate, "force_flush", return_value=True) as mock_ff:
            result = exp.force_flush(5000)
        mock_ff.assert_called_once_with(5000)
        assert result is True
