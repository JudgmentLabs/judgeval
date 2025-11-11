from __future__ import annotations

from typing import Optional

from opentelemetry.context import Context, get_value
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from judgeval.v1.tracer.processors._lifecycles.registry import register
from judgeval.v1.tracer.processors._lifecycles.context_keys import (
    AGENT_ID_KEY,
    PARENT_AGENT_ID_KEY,
    AGENT_CLASS_NAME_KEY,
    AGENT_INSTANCE_NAME_KEY,
)
from judgeval.judgment_attribute_keys import AttributeKeys


class AgentIdProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        agent_id = get_value(AGENT_ID_KEY, context=parent_context)
        if agent_id is not None:
            span.set_attribute(AttributeKeys.JUDGMENT_AGENT_ID, str(agent_id))

        parent_agent_id = get_value(PARENT_AGENT_ID_KEY, context=parent_context)
        if parent_agent_id is not None:
            span.set_attribute(
                AttributeKeys.JUDGMENT_PARENT_AGENT_ID, str(parent_agent_id)
            )

        class_name = get_value(AGENT_CLASS_NAME_KEY, context=parent_context)
        if class_name is not None:
            span.set_attribute(AttributeKeys.JUDGMENT_AGENT_CLASS_NAME, str(class_name))

        instance_name = get_value(AGENT_INSTANCE_NAME_KEY, context=parent_context)
        if instance_name is not None:
            span.set_attribute(
                AttributeKeys.JUDGMENT_AGENT_INSTANCE_NAME, str(instance_name)
            )

        if agent_id is not None and agent_id != parent_agent_id:
            span.set_attribute(AttributeKeys.JUDGMENT_IS_AGENT_ENTRY_POINT, True)

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


register(AgentIdProcessor)
