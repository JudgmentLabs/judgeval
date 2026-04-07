from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span
from opentelemetry.util.types import Attributes

from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

if TYPE_CHECKING:
    from judgeval.trace.tracer import Tracer


@dataclass(frozen=True)
class LinkedSubagentSpans:
    invocation_span: Span
    child_span: Span


class SubagentManager:
    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    @contextmanager
    def start_linked_root_span(
        self,
        name: str,
        source_span: Span,
        attributes: Attributes = None,
        *,
        end_on_exit: bool = True,
    ) -> Iterator[LinkedSubagentSpans]:
        source_ctx = source_span.get_span_context()
        if not source_ctx.is_valid:
            raise RuntimeError(
                "start_subagent_span() requires a valid parent span context."
            )

        proxy = JudgmentTracerProvider.get_instance()
        invocation_span = proxy.get_tracer(
            JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
        ).start_span(name)
        try:
            invocation_ctx = invocation_span.get_span_context()
            if not invocation_ctx.is_valid:
                raise RuntimeError(
                    "Failed to create parent-side subagent invocation span."
                )
            invocation_span.set_attribute(AttributeKeys.JUDGMENT_SPAN_KIND, "agent")

            child_attributes = dict(attributes or {})
            child_attributes[AttributeKeys.JUDGMENT_SPAN_KIND] = "agent"
            child_attributes[AttributeKeys.JUDGMENT_LINK_SOURCE_TRACE_ID] = format(
                invocation_ctx.trace_id, "032x"
            )
            child_attributes[AttributeKeys.JUDGMENT_LINK_SOURCE_SPAN_ID] = format(
                invocation_ctx.span_id, "016x"
            )

            child_span = proxy.get_tracer(
                JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
            ).start_span(
                name,
                context=trace_api.set_span_in_context(
                    trace_api.INVALID_SPAN,
                    proxy.get_current_context(),
                ),
                attributes=child_attributes,
                links=[trace_api.Link(invocation_ctx)],
            )
            child_ctx = child_span.get_span_context()

            if (
                invocation_span.is_recording()
                and child_span.is_recording()
                and child_ctx.is_valid
            ):
                invocation_span.set_attribute(
                    AttributeKeys.JUDGMENT_LINK_TARGET_TRACE_ID,
                    format(child_ctx.trace_id, "032x"),
                )
                invocation_span.set_attribute(
                    AttributeKeys.JUDGMENT_LINK_TARGET_SPAN_ID,
                    format(child_ctx.span_id, "016x"),
                )
                invocation_span.add_link(child_ctx)

            with proxy.use_span(child_span, end_on_exit=end_on_exit) as span:
                yield LinkedSubagentSpans(
                    invocation_span=invocation_span,
                    child_span=span,
                )
        finally:
            if invocation_span.is_recording():
                invocation_span.end()
