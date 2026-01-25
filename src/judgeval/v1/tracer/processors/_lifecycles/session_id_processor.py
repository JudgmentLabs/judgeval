from __future__ import annotations

from typing import Optional

from opentelemetry.context import Context, get_value
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from judgeval.v1.tracer.processors._lifecycles.registry import register
from judgeval.v1.tracer.processors._lifecycles.context_keys import SESSION_ID_KEY
from judgeval.judgment_attribute_keys import AttributeKeys


class SessionIdProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        session_id = get_value(SESSION_ID_KEY, context=parent_context)
        if session_id is not None:
            span.set_attribute(AttributeKeys.JUDGMENT_SESSION_ID, str(session_id))

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


register(SessionIdProcessor)
