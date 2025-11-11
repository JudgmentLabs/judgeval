from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from collections import defaultdict

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.trace import get_current_span
from opentelemetry.trace.span import SpanContext
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
)

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.tracer.keys import InternalAttributeKeys
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.v1.tracer.processors._lifecycles import get_all


if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer


class JudgmentSpanProcessor(BatchSpanProcessor):
    __slots__ = ("tracer", "resource_attributes", "_internal_attributes")

    def __init__(
        self,
        tracer: BaseTracer,
        exporter: SpanExporter,
        /,
        *,
        max_queue_size: int | None = None,
        schedule_delay_millis: float | None = None,
        max_export_batch_size: int | None = None,
        export_timeout_millis: float | None = None,
    ):
        self.tracer = tracer

        super().__init__(
            exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_millis,
            max_export_batch_size=max_export_batch_size,
            export_timeout_millis=export_timeout_millis,
        )
        self._internal_attributes: defaultdict[tuple[int, int], dict[str, Any]] = (
            defaultdict(dict)
        )

    def _get_span_key(self, span_context: SpanContext) -> tuple[int, int]:
        return (span_context.trace_id, span_context.span_id)

    def set_internal_attribute(
        self, span_context: SpanContext, key: str, value: Any
    ) -> None:
        span_key = self._get_span_key(span_context)
        self._internal_attributes[span_key][key] = value

    def get_internal_attribute(
        self, span_context: SpanContext, key: str, default: Any = None
    ) -> Any:
        span_key = self._get_span_key(span_context)
        return self._internal_attributes[span_key].get(key, default)

    def increment_update_id(self, span_context: SpanContext) -> int:
        current_id = self.get_internal_attribute(
            span_context=span_context, key=AttributeKeys.JUDGMENT_UPDATE_ID, default=0
        )
        new_id = current_id + 1
        self.set_internal_attribute(
            span_context=span_context,
            key=AttributeKeys.JUDGMENT_UPDATE_ID,
            value=new_id,
        )
        return current_id

    def _cleanup_span_state(self, span_key: tuple[int, int]) -> None:
        self._internal_attributes.pop(span_key, None)

    @dont_throw
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        for processor in get_all():
            processor.on_start(span, parent_context)

    @dont_throw
    def emit_partial(self) -> None:
        current_span = get_current_span()
        if (
            not current_span
            or not current_span.is_recording()
            or not isinstance(current_span, ReadableSpan)
        ):
            return

        span_context = current_span.get_span_context()
        if self.get_internal_attribute(
            span_context, InternalAttributeKeys.DISABLE_PARTIAL_EMIT, False
        ):
            return

        attributes = dict(current_span.attributes or {})
        attributes[AttributeKeys.JUDGMENT_UPDATE_ID] = self.increment_update_id(
            span_context
        )

        partial_span = ReadableSpan(
            name=current_span.name,
            context=span_context,
            parent=current_span.parent,
            resource=current_span.resource,
            attributes=attributes,
            events=current_span.events,
            links=current_span.links,
            status=current_span.status,
            kind=current_span.kind,
            start_time=current_span.start_time,
            end_time=None,
            instrumentation_scope=current_span.instrumentation_scope,
        )

        super().on_end(partial_span)

    @dont_throw
    def on_end(self, span: ReadableSpan) -> None:
        for processor in get_all():
            processor.on_end(span)

        if not span.context:
            super().on_end(span)
            return

        span_key = self._get_span_key(span.context)

        if self.get_internal_attribute(
            span.context, InternalAttributeKeys.CANCELLED, False
        ):
            self._cleanup_span_state(span_key)
            return

        if span.end_time is not None:
            attributes = dict(span.attributes or {})
            attributes[AttributeKeys.JUDGMENT_UPDATE_ID] = 20

            final_span = ReadableSpan(
                name=span.name,
                context=span.context,
                parent=span.parent,
                resource=span.resource,
                attributes=attributes,
                events=span.events,
                links=span.links,
                status=span.status,
                kind=span.kind,
                start_time=span.start_time,
                end_time=span.end_time,
                instrumentation_scope=span.instrumentation_scope,
            )

            self._cleanup_span_state(span_key)
            super().on_end(final_span)
        else:
            super().on_end(span)
