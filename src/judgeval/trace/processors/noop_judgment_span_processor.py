from __future__ import annotations

from typing import Any, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.trace.span import SpanContext

from judgeval.trace.processors.judgment_span_processor import JudgmentSpanProcessor
from judgeval.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)


class NoOpJudgmentSpanProcessor(JudgmentSpanProcessor):
    """A span processor that silently discards all operations.

    Used internally when monitoring is disabled. All state operations
    return safe defaults and no spans are exported.
    """

    __slots__ = ()

    def __init__(self):
        pass

    @property
    def span_exporter(self) -> NoOpJudgmentSpanExporter:
        return NoOpJudgmentSpanExporter()

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int | None = 30000) -> bool:
        return True

    def emit_partial(self) -> None:
        pass

    def state_set(self, span_context: SpanContext, key: str, value: Any) -> None:
        pass

    def state_get(
        self, span_context: SpanContext, key: str, default: Any = None
    ) -> Any:
        return default

    def state_incr(self, span_context: SpanContext, key: str) -> int:
        return 0

    def state_append(self, span_context: SpanContext, key: str, item: Any) -> list[Any]:
        return []
