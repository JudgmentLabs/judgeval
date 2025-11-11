from __future__ import annotations

from typing import Optional

from opentelemetry.context import Context, get_value
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from judgeval.v1.tracer.processors._lifecycles.registry import register
from judgeval.v1.tracer.processors._lifecycles.context_keys import CUSTOMER_ID_KEY
from judgeval.judgment_attribute_keys import AttributeKeys


class CustomerIdProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        customer_id = get_value(CUSTOMER_ID_KEY, context=parent_context)
        if customer_id is not None:
            span.set_attribute(AttributeKeys.JUDGMENT_CUSTOMER_ID, str(customer_id))

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


register(CustomerIdProcessor)
