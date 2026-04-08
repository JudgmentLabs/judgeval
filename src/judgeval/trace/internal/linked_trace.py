from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span
from opentelemetry.util.types import Attributes

from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.logger import judgeval_logger
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider


@dataclass(frozen=True, slots=True)
class LinkedTraceSpans:
    invocation_span: Span
    linked_root_span: Span


def _noop_linked_trace_spans(
    invocation_span: Span = trace_api.INVALID_SPAN,
) -> LinkedTraceSpans:
    return LinkedTraceSpans(
        invocation_span=invocation_span,
        linked_root_span=trace_api.INVALID_SPAN,
    )


@contextmanager
def start_linked_trace(
    name: str,
    source_span: Span,
    attributes: Attributes = None,
    *,
    span_type: Optional[str] = "span",
    end_on_exit: bool = True,
    end_invocation_on_exit: bool = True,
) -> Iterator[LinkedTraceSpans]:
    source_ctx = source_span.get_span_context()
    if not source_ctx.is_valid:
        judgeval_logger.warning(
            "start_linked_trace() received an invalid parent span context. "
            "Continuing without linked-trace tracing."
        )
        yield _noop_linked_trace_spans()
        return

    proxy = JudgmentTracerProvider.get_instance()
    invocation_span = proxy.get_tracer(
        JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
    ).start_span(name)
    try:
        invocation_ctx = invocation_span.get_span_context()
        if not invocation_ctx.is_valid:
            judgeval_logger.warning(
                "Failed to create a valid parent-side linked-trace invocation span. "
                "Continuing without linked-trace tracing."
            )
            yield _noop_linked_trace_spans()
            return
        if span_type:
            invocation_span.set_attribute(AttributeKeys.JUDGMENT_SPAN_KIND, span_type)

        linked_root_attributes = dict(attributes or {})
        if span_type:
            linked_root_attributes[AttributeKeys.JUDGMENT_SPAN_KIND] = span_type
        linked_root_attributes[AttributeKeys.JUDGMENT_LINK_SOURCE_TRACE_ID] = format(
            invocation_ctx.trace_id, "032x"
        )
        linked_root_attributes[AttributeKeys.JUDGMENT_LINK_SOURCE_SPAN_ID] = format(
            invocation_ctx.span_id, "016x"
        )

        try:
            linked_root_span = proxy.get_tracer(
                JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
            ).start_span(
                name,
                context=trace_api.set_span_in_context(
                    trace_api.INVALID_SPAN,
                    proxy.get_current_context(),
                ),
                attributes=linked_root_attributes,
                links=[trace_api.Link(invocation_ctx)],
            )
        except Exception as e:
            judgeval_logger.warning(
                "Failed to create linked-trace root span '%s': %s. "
                "Continuing without linked-trace tracing.",
                name,
                e,
            )
            yield _noop_linked_trace_spans(invocation_span)
            return

        linked_root_ctx = linked_root_span.get_span_context()
        if not linked_root_ctx.is_valid:
            judgeval_logger.warning(
                "Failed to create a valid linked-trace root span for '%s'. "
                "Continuing without linked-trace tracing.",
                name,
            )
            yield _noop_linked_trace_spans(invocation_span)
            return

        if (
            invocation_span.is_recording()
            and linked_root_span.is_recording()
            and linked_root_ctx.is_valid
        ):
            invocation_span.set_attribute(
                AttributeKeys.JUDGMENT_LINK_TARGET_TRACE_ID,
                format(linked_root_ctx.trace_id, "032x"),
            )
            invocation_span.set_attribute(
                AttributeKeys.JUDGMENT_LINK_TARGET_SPAN_ID,
                format(linked_root_ctx.span_id, "016x"),
            )
            invocation_span.add_link(linked_root_ctx)

        with proxy.use_span(linked_root_span, end_on_exit=end_on_exit) as span:
            yield LinkedTraceSpans(
                invocation_span=invocation_span,
                linked_root_span=span,
            )
    finally:
        if end_invocation_on_exit and invocation_span.is_recording():
            invocation_span.end()
