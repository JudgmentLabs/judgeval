from __future__ import annotations

from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from judgeval.trace.exporters.judgment_span_exporter import JudgmentSpanExporter


class NoOpJudgmentSpanExporter(JudgmentSpanExporter):
    """A span exporter that silently discards all spans.

    Used internally when monitoring is disabled (e.g. missing credentials).
    All operations succeed immediately without performing any I/O.
    """

    __slots__ = ()

    def __init__(self):
        pass

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
