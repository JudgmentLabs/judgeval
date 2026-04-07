from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as sdk_trace
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

    def _allocate_trace_id(self) -> int:
        sdk_tracer = self._get_sdk_tracer()
        return sdk_tracer.id_generator.generate_trace_id()

    def _allocate_span_id(self) -> int:
        sdk_tracer = self._get_sdk_tracer()
        return sdk_tracer.id_generator.generate_span_id()

    def _get_sdk_tracer(self) -> sdk_trace.Tracer:
        sdk_tracer = self._tracer._tracer_provider.get_tracer(
            JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
        )
        if not isinstance(sdk_tracer, sdk_trace.Tracer):
            raise RuntimeError("Active tracer does not expose an SDK tracer.")
        return sdk_tracer

    def _clear_current_span_from_context(self):
        proxy = JudgmentTracerProvider.get_instance()
        return trace_api.set_span_in_context(
            trace_api.INVALID_SPAN, proxy.get_current_context()
        )

    def _start_root_span_with_ids(
        self,
        name: str,
        trace_id: int,
        span_id: int,
        *,
        attributes: Attributes = None,
        links: Optional[Sequence[trace_api.Link]] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        sdk_tracer = self._get_sdk_tracer()
        parentless_context = self._clear_current_span_from_context()
        parent_span_context = trace_api.get_current_span(
            parentless_context
        ).get_span_context()
        if parent_span_context is not None and parent_span_context.is_valid:
            raise RuntimeError(
                "Explicit root spans must not have a valid parent span context."
            )

        span_links = tuple(links or ())
        sampling_result = sdk_tracer.sampler.should_sample(
            parentless_context,
            trace_id,
            name,
            trace_api.SpanKind.INTERNAL,
            attributes,
            span_links,
        )

        trace_flags = (
            trace_api.TraceFlags(trace_api.TraceFlags.SAMPLED)
            if sampling_result.decision.is_sampled()
            else trace_api.TraceFlags(trace_api.TraceFlags.DEFAULT)
        )
        span_context = trace_api.SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=trace_flags,
            trace_state=sampling_result.trace_state,
        )

        if not sampling_result.decision.is_recording():
            return trace_api.NonRecordingSpan(context=span_context)

        span = sdk_trace._Span(
            name=name,
            context=span_context,
            parent=None,
            sampler=sdk_tracer.sampler,
            resource=sdk_tracer.resource,
            attributes=sampling_result.attributes.copy(),
            span_processor=sdk_tracer.span_processor,
            kind=trace_api.SpanKind.INTERNAL,
            links=span_links,
            instrumentation_info=sdk_tracer.instrumentation_info,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            limits=sdk_tracer._span_limits,
            instrumentation_scope=sdk_tracer._instrumentation_scope,
        )
        span.start(start_time=start_time, parent_context=parentless_context)
        return span

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
        invocation_ctx = invocation_span.get_span_context()
        if not invocation_ctx.is_valid:
            invocation_span.end()
            raise RuntimeError("Failed to create parent-side subagent invocation span.")
        invocation_span.set_attribute(AttributeKeys.JUDGMENT_SPAN_KIND, "agent")

        child_trace_id = self._allocate_trace_id()
        child_span_id = self._allocate_span_id()

        child_attributes = dict(attributes or {})
        child_attributes[AttributeKeys.JUDGMENT_SPAN_KIND] = "agent"
        child_attributes[AttributeKeys.JUDGMENT_LINK_SOURCE_TRACE_ID] = format(
            invocation_ctx.trace_id, "032x"
        )
        child_attributes[AttributeKeys.JUDGMENT_LINK_SOURCE_SPAN_ID] = format(
            invocation_ctx.span_id, "016x"
        )

        child_span = self._start_root_span_with_ids(
            name,
            child_trace_id,
            child_span_id,
            attributes=child_attributes,
            links=[trace_api.Link(invocation_ctx)],
        )

        try:
            if invocation_span.is_recording() and child_span.is_recording():
                invocation_span.set_attribute(
                    AttributeKeys.JUDGMENT_LINK_TARGET_TRACE_ID,
                    format(child_trace_id, "032x"),
                )
                invocation_span.set_attribute(
                    AttributeKeys.JUDGMENT_LINK_TARGET_SPAN_ID,
                    format(child_span_id, "016x"),
                )

            with proxy.use_span(child_span, end_on_exit=end_on_exit) as span:
                yield LinkedSubagentSpans(
                    invocation_span=invocation_span,
                    child_span=span,
                )
        finally:
            if invocation_span.is_recording():
                invocation_span.end()
