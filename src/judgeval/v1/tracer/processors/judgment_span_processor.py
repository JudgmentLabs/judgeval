from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional
from weakref import finalize

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
    __slots__ = (
        "tracer",
        "resource_attributes",
        "_span_finalizers",
        "_internal_attributes",
        "_lock",
    )

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

        self._lock = threading.RLock()
        self._span_finalizers: dict[tuple[int, int], finalize] = {}
        self._internal_attributes: dict[tuple[int, int], dict[str, Any]] = {}

    def _cleanup_span_state(self, span_key: tuple[int, int]) -> None:
        with self._lock:
            self._internal_attributes.pop(span_key, None)
            self._span_finalizers.pop(span_key, None)

    def _register_span(self, span: Span) -> None:
        if not span.context:
            return

        span_key = (span.context.trace_id, span.context.span_id)

        with self._lock:
            self._span_finalizers[span_key] = finalize(
                span, self._cleanup_span_state, span_key
            )

    def set_internal_attribute(
        self, span_context: SpanContext, key: str, value: Any
    ) -> None:
        span_key = (span_context.trace_id, span_context.span_id)
        with self._lock:
            if span_key not in self._internal_attributes:
                self._internal_attributes[span_key] = {}
            self._internal_attributes[span_key][key] = value

    def get_internal_attribute(
        self, span_context: SpanContext, key: str, default: Any = None
    ) -> Any:
        span_key = (span_context.trace_id, span_context.span_id)
        with self._lock:
            return self._internal_attributes.get(span_key, {}).get(key, default)

    def _emit_span(self, span: ReadableSpan) -> None:
        if not span.context:
            return

        span_key = (span.context.trace_id, span.context.span_id)

        with self._lock:
            internal_attrs = self._internal_attributes.setdefault(span_key, {})
            curr_id = internal_attrs.get(AttributeKeys.JUDGMENT_UPDATE_ID, 0)
            internal_attrs[AttributeKeys.JUDGMENT_UPDATE_ID] = curr_id + 1

        attributes = dict(span.attributes or {}) | {
            AttributeKeys.JUDGMENT_UPDATE_ID: curr_id
        }

        emitted_span = ReadableSpan(
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
            end_time=span.end_time or span.start_time,
            instrumentation_scope=span.instrumentation_scope,
        )

        super().on_end(emitted_span)

    @dont_throw
    def emit_partial(self) -> None:
        span = get_current_span()
        if (
            not span.is_recording()
            or not isinstance(span, ReadableSpan)
            or not span.context
            or self.get_internal_attribute(
                span.context, InternalAttributeKeys.DISABLE_PARTIAL_EMIT, False
            )
        ):
            return

        self._emit_span(span=span)

    @dont_throw
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        for processor in get_all():
            processor.on_start(span, parent_context)

        # Register span for weak reference tracking with cleanup callback
        self._register_span(span)

    @dont_throw
    def on_end(self, span: ReadableSpan) -> None:
        for processor in get_all():
            processor.on_end(span)

        if not span.context:
            super().on_end(span)
            return

        is_cancelled = self.get_internal_attribute(
            span.context, InternalAttributeKeys.CANCELLED, False
        )
        if not is_cancelled:
            self._emit_span(span=span)

        span_key = (span.context.trace_id, span.context.span_id)
        self._cleanup_span_state(span_key)
