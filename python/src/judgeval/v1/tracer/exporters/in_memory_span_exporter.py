from __future__ import annotations

from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from judgeval.v1.tracer.exporters.span_store import ABCSpanStore


class InMemorySpanExporter(SpanExporter):
    __slots__ = ("_store",)

    def __init__(self, store: ABCSpanStore) -> None:
        self._store = store

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._store.add(*spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
