from __future__ import annotations

from typing import Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from judgeval.logger import judgeval_logger


class JudgmentSpanExporter(SpanExporter):
    __slots__ = ("_delegate",)

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        organization_id: str,
        project_id: str,
    ):
        self._delegate = OTLPSpanExporter(
            endpoint=endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Organization-Id": organization_id,
                "X-Project-Id": project_id,
            },
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        judgeval_logger.info(f"Exported {len(spans)} spans")
        return self._delegate.export(spans)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)
