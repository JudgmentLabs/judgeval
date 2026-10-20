from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import requests

from opentelemetry.sdk.trace.export import SpanExportResult

from judgeval.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)
from judgeval.trace.exporters.judgment_span_exporter import JudgmentSpanExporter


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

    @pytest.mark.parametrize(
        "body, message",
        [
            (
                b'{"error":"Usage limit exceeded","message":"Allowance exhausted"}',
                "Allowance exhausted",
            ),
            (
                b"<html>Payment Required</html>",
                "Payment required. Review your organization's billing and usage limits.",
            ),
            (
                b'{"message":123}',
                "Payment required. Review your organization's billing and usage limits.",
            ),
        ],
    )
    def test_billing_diagnostics_through_http_export(self, body, message):
        exporter = self._make_exporter()
        responses = []
        for status in [402, 402, 200, 402, 403]:
            response = requests.Response()
            response.status_code = status
            response.reason = "Payment Required" if status == 402 else "OK"
            response._content = body
            responses.append(response)
        try:
            with (
                patch("requests.adapters.HTTPAdapter.send", side_effect=responses),
                patch(
                    "judgeval.trace.exporters.judgment_span_exporter.judgeval_logger.error"
                ) as log,
            ):
                results = [exporter.export([]) for _ in responses]

            assert (results, log.call_args_list) == (
                [
                    SpanExportResult.FAILURE,
                    SpanExportResult.FAILURE,
                    SpanExportResult.SUCCESS,
                    SpanExportResult.FAILURE,
                    SpanExportResult.FAILURE,
                ],
                [call("Trace export rejected (HTTP 402): %s", message)] * 2,
            )
        finally:
            exporter.shutdown()
