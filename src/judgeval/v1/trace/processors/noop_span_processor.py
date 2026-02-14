from __future__ import annotations

from typing import Any, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace.span import SpanContext


class NoOpSpanProcessor(SpanProcessor):
    __slots__ = ()

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

    def set_internal_attribute(
        self, span_context: SpanContext, key: str, value: Any
    ) -> None:
        pass

    def get_internal_attribute(
        self, span_context: SpanContext, key: str, default: Any = None
    ) -> Any:
        return default
