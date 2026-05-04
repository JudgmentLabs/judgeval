from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from weakref import finalize

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace.span import SpanContext

from judgeval.judgment_attribute_keys import AttributeKeys, InternalAttributeKeys
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.trace.processors.judgment_baggage_processor import (
    JudgmentBaggageProcessor,
)

if TYPE_CHECKING:
    from judgeval.data.example import Example
    from judgeval.trace.base_tracer import BaseTracer


class OfflineJudgmentSpanProcessor(SimpleSpanProcessor):
    """Synchronous span processor used by :class:`OfflineTracer`.

    Unlike :class:`JudgmentSpanProcessor` (which extends
    ``BatchSpanProcessor`` and exports spans on a background worker
    thread), this processor exports each span synchronously via the
    configured exporter as soon as it ends. That matches the
    short-lived, deterministic lifecycle of offline / experiment runs
    where you want the trace flushed before moving on to the next
    example.

    Per-span state management (counters, lists), partial-span emission,
    and baggage propagation are implemented locally instead of being
    inherited from the batched ``JudgmentSpanProcessor`` so that nothing
    in this class spawns background threads.

    On every *root* span end this processor also appends a new
    :class:`Example` to the caller-supplied ``dataset`` list, populated
    with the static ``example_fields`` plus an ``offline_trace_id`` field
    referencing the offline trace.
    """

    __slots__ = (
        "tracer",
        "_dataset",
        "_example_fields",
        "_dataset_lock",
        "_seen_trace_ids",
        "_state_lock",
        "_state",
        "_span_finalizers",
        "_baggage_processor",
    )

    def __init__(
        self,
        tracer: BaseTracer,
        exporter: SpanExporter,
        /,
        *,
        dataset: List[Example],
        example_fields: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(exporter)
        self.tracer = tracer
        self._dataset = dataset
        self._example_fields: Dict[str, Any] = dict(example_fields or {})
        self._dataset_lock = threading.Lock()
        self._seen_trace_ids: set[str] = set()

        self._state_lock = threading.RLock()
        self._state: dict[tuple[int, int], dict[str, Any]] = {}
        self._span_finalizers: dict[tuple[int, int], finalize] = {}
        self._baggage_processor = JudgmentBaggageProcessor()

    # ------------------------------------------------------------------ #
    #  State management (mirrors JudgmentSpanProcessor's API)            #
    # ------------------------------------------------------------------ #

    def _cleanup_span_state(self, span_key: tuple[int, int]) -> None:
        with self._state_lock:
            self._state.pop(span_key, None)
            self._span_finalizers.pop(span_key, None)

    def _register_span(self, span: Span) -> None:
        if not span.context:
            return
        span_key = (span.context.trace_id, span.context.span_id)
        with self._state_lock:
            self._span_finalizers[span_key] = finalize(
                span, self._cleanup_span_state, span_key
            )

    def state_set(self, span_context: SpanContext, key: str, value: Any) -> None:
        """Store a value in the mutable state for a span."""
        span_key = (span_context.trace_id, span_context.span_id)
        with self._state_lock:
            self._state.setdefault(span_key, {})[key] = value

    def state_get(
        self, span_context: SpanContext, key: str, default: Any = None
    ) -> Any:
        """Retrieve a value from the mutable state for a span."""
        span_key = (span_context.trace_id, span_context.span_id)
        with self._state_lock:
            return self._state.get(span_key, {}).get(key, default)

    def state_incr(self, span_context: SpanContext, key: str) -> int:
        """Atomically increment a counter. Returns the value before increment."""
        span_key = (span_context.trace_id, span_context.span_id)
        with self._state_lock:
            attrs = self._state.setdefault(span_key, {})
            stored = attrs.get(key, 0)
            prev: int = stored if isinstance(stored, int) else 0
            attrs[key] = prev + 1
            return prev

    def state_append(self, span_context: SpanContext, key: str, item: Any) -> list[Any]:
        """Atomically append to a list. Returns the new list."""
        span_key = (span_context.trace_id, span_context.span_id)
        with self._state_lock:
            attrs = self._state.setdefault(span_key, {})
            stored = attrs.get(key, [])
            lst: list[Any] = [*(stored if isinstance(stored, list) else []), item]
            attrs[key] = lst
            return lst

    # ------------------------------------------------------------------ #
    #  Span lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def _emit_span(self, span: ReadableSpan, *, is_partial: bool = False) -> None:
        if not span.context:
            return
        curr_id = self.state_incr(span.context, AttributeKeys.JUDGMENT_UPDATE_ID)
        attributes = dict(span.attributes or {}) | {
            AttributeKeys.JUDGMENT_UPDATE_ID: curr_id
        }

        if is_partial:
            attributes.pop(AttributeKeys.JUDGMENT_PENDING_TRACE_EVAL, None)

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
        # SimpleSpanProcessor.on_end exports the span synchronously.
        super().on_end(emitted_span)

    @dont_throw
    def emit_partial(self) -> None:
        """Synchronously export the current span's in-progress state."""
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
        self._emit_span(span=span, is_partial=True)

    @dont_throw
    def _maybe_create_example(self, span: ReadableSpan) -> None:
        if span.parent is not None or not span.context:
            return

        trace_id_hex = format(span.context.trace_id, "032x")

        with self._dataset_lock:
            if trace_id_hex in self._seen_trace_ids:
                return
            self._seen_trace_ids.add(trace_id_hex)

        from judgeval.data.example import Example

        properties: Dict[str, Any] = {
            **self._example_fields,
            "offline_trace_id": trace_id_hex,
        }
        example = Example.create(**properties)

        with self._dataset_lock:
            self._dataset.append(example)

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
                self._maybe_create_example(span)
                self._emit_span(span=span)
        finally:
            self._cleanup_span_state(span_key)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush: ``SimpleSpanProcessor`` exports synchronously."""
        return True
