from opentelemetry.sdk.trace import Span, ReadableSpan
from opentelemetry.context import Context
from typing import Any, Optional

from opentelemetry.trace import SpanContext

from judgeval.v1.tracer.processors.judgment_span_processor import JudgmentSpanProcessor


class NoOpJudgmentSpanProcessor(JudgmentSpanProcessor):
    __slots__ = ("resource_attributes",)

    def __init__(self):
        self.resource_attributes = {}

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

    def increment_update_id(self, span_context: SpanContext) -> int:
        return 0
