"""
Custom OpenTelemetry exporter for Judgment API.

This exporter sends spans to the Judgment API using the existing format.
The BatchSpanProcessor handles all batching, threading, and retry logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan

from judgeval.common.tracer.span_transformer import SpanTransformer
from judgeval.common.logger import judgeval_logger
from judgeval.common.api.api import JudgmentApiClient


class JudgmentAPISpanExporter(SpanExporter):
    """
    Custom OpenTelemetry exporter that sends spans to Judgment API.

    This exporter is used by BatchSpanProcessor which handles all the
    batching, threading, and retry logic for us.
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
    ):
        self.api_client = JudgmentApiClient(judgment_api_key, organization_id)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans to Judgment API.

        This method is called by BatchSpanProcessor with a batch of spans.
        We send them synchronously since BatchSpanProcessor handles threading.
        """
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            spans_data = []
            eval_runs_data = []

            for span in spans:
                span_data = self._convert_span_to_judgment_format(span)

                if span.attributes and span.attributes.get("judgment.evaluation_run"):
                    eval_runs_data.append(span_data)
                else:
                    spans_data.append(span_data)

            if spans_data:
                self._send_spans_batch(spans_data)

            if eval_runs_data:
                self._send_evaluation_runs_batch(eval_runs_data)

            return SpanExportResult.SUCCESS

        except Exception as e:
            judgeval_logger.error(f"Error in JudgmentAPISpanExporter.export: {e}")
            return SpanExportResult.FAILURE

    def _convert_span_to_judgment_format(self, span: ReadableSpan) -> Dict[str, Any]:
        """Convert OpenTelemetry span to existing Judgment API format."""
        if span.attributes and span.attributes.get("judgment.evaluation_run"):
            return SpanTransformer.otel_span_to_evaluation_run_format(span)
        else:
            return SpanTransformer.otel_span_to_judgment_format(span)

    def _send_spans_batch(self, spans: List[Dict[str, Any]]):
        """Send a batch of spans to the spans endpoint."""
        spans_data = [span["data"] for span in spans]
        self.api_client.send_spans_batch(spans_data)

    def _send_evaluation_runs_batch(self, eval_runs: List[Dict[str, Any]]):
        """Send a batch of evaluation runs to the evaluation runs endpoint."""
        evaluation_entries = []
        for eval_run in eval_runs:
            eval_data = eval_run["data"]
            entry = {
                "evaluation_run": {
                    key: value
                    for key, value in eval_data.items()
                    if key not in ["associated_span_id", "span_data", "queued_at"]
                },
                "associated_span": {
                    "span_id": eval_data.get("associated_span_id"),
                    "span_data": eval_data.get("span_data"),
                },
                "queued_at": eval_data.get("queued_at"),
            }
            evaluation_entries.append(entry)

        self.api_client.send_evaluation_runs_batch(evaluation_entries)

    def shutdown(self, timeout_millis: int = 30000) -> None:
        """Shutdown the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending requests."""
        return True
