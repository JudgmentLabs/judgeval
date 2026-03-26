from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional
from weakref import finalize

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.trace.span import SpanContext
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from judgeval.judgment_attribute_keys import AttributeKeys, InternalAttributeKeys
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.trace.processors.judgment_baggage_processor import (
    JudgmentBaggageProcessor,
)

if TYPE_CHECKING:
    from judgeval.trace.base_tracer import BaseTracer


class JudgmentSpanProcessor(BatchSpanProcessor):
    """Span processor that manages span lifecycle, state, and batched export.

    Extends the OpenTelemetry ``BatchSpanProcessor`` with per-span mutable
    state (counters, lists), partial-span emission for streaming updates,
    and automatic baggage propagation.

    Created automatically by ``Tracer.init()``. Use it directly only when
    building a custom tracing pipeline.

    Args:
        tracer: The ``BaseTracer`` instance that owns this processor.
        exporter: The span exporter to send completed spans to.
        max_queue_size: Maximum number of spans queued before dropping.
        schedule_delay_millis: Delay between export batches in milliseconds.
        max_export_batch_size: Maximum spans per export batch.
        export_timeout_millis: Timeout for each export call in milliseconds.
    """

    __slots__ = (
        "tracer",
        "_span_finalizers",
        "_state",
        "_lock",
        "_baggage_processor",
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
        self._state: dict[tuple[int, int], dict[str, Any]] = {}
        self._baggage_processor = JudgmentBaggageProcessor()

    def _cleanup_span_state(self, span_key: tuple[int, int]) -> None:
        with self._lock:
            self._state.pop(span_key, None)
            self._span_finalizers.pop(span_key, None)

    def _register_span(self, span: Span) -> None:
        if not span.context:
            return
        span_key = (span.context.trace_id, span.context.span_id)
        with self._lock:
            self._span_finalizers[span_key] = finalize(
                span, self._cleanup_span_state, span_key
            )

    def state_set(self, span_context: SpanContext, key: str, value: Any) -> None:
        """Store a value in the mutable state for a span."""
        span_key = (span_context.trace_id, span_context.span_id)
        with self._lock:
            self._state.setdefault(span_key, {})[key] = value

    def state_get(
        self, span_context: SpanContext, key: str, default: Any = None
    ) -> Any:
        """Retrieve a value from the mutable state for a span."""
        span_key = (span_context.trace_id, span_context.span_id)
        with self._lock:
            return self._state.get(span_key, {}).get(key, default)

    def state_incr(self, span_context: SpanContext, key: str) -> int:
        """Atomically increment a counter. Returns the value before increment."""
        span_key = (span_context.trace_id, span_context.span_id)
        with self._lock:
            attrs = self._state.setdefault(span_key, {})
            stored = attrs.get(key, 0)
            prev: int = stored if isinstance(stored, int) else 0
            attrs[key] = prev + 1
            return prev

    def state_append(self, span_context: SpanContext, key: str, item: Any) -> list[Any]:
        """Atomically append to a list. Returns the new list."""
        span_key = (span_context.trace_id, span_context.span_id)
        with self._lock:
            attrs = self._state.setdefault(span_key, {})
            stored = attrs.get(key, [])
            lst: list[Any] = [*(stored if isinstance(stored, list) else []), item]
            attrs[key] = lst
            return lst

    def _emit_span(self, span: ReadableSpan) -> None:
        if not span.context:
            return
        curr_id = self.state_incr(span.context, AttributeKeys.JUDGMENT_UPDATE_ID)
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
        """Export the current span's in-progress state for streaming updates."""
        from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

        proxy = JudgmentTracerProvider.get_instance()
        span = proxy.get_current_span()
        if (
            not span.is_recording()
            or not isinstance(span, ReadableSpan)
            or not span.context
            or self.state_get(
                span.context, InternalAttributeKeys.DISABLE_PARTIAL_EMIT, False
            )
        ):
            return
        self._emit_span(span=span)

    @dont_throw
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self._baggage_processor.on_start(span, parent_context)
        self._register_span(span)

    @dont_throw
    def on_end(self, span: ReadableSpan) -> None:
        if not span.context:
            super().on_end(span)
            return
        span_key = (span.context.trace_id, span.context.span_id)
        try:
            is_cancelled = self.state_get(
                span.context, InternalAttributeKeys.CANCELLED, False
            )
            if not is_cancelled:
                self._emit_span(span=span)
        finally:
            self._cleanup_span_state(span_key)
