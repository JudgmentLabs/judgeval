from opentelemetry.sdk.trace.export import (
    SpanExportResult,
    SpanExporter,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from typing import Sequence

from judgeval.tracer.exporters.store import ABCSpanStore


class JudgmentSpanExporter(OTLPSpanExporter):
    def __init__(self, endpoint: str, api_key: str, organization_id: str):
        super().__init__(
            endpoint=endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Organization-Id": organization_id,
            },
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return super().export(spans)


class InMemorySpanExporter(SpanExporter):
    def __init__(self, store: ABCSpanStore):
        self.store = store

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.store.add(*spans)
        return SpanExportResult.SUCCESS


__all__ = ("JudgmentSpanExporter", "InMemorySpanExporter")
