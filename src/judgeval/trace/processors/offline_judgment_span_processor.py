from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace.span import SpanContext

from judgeval.trace.processors.judgment_baggage_processor import (
    JudgmentBaggageProcessor,
)
from judgeval.utils.decorators.dont_throw import dont_throw

if TYPE_CHECKING:
    from judgeval.data.example import Example
    from judgeval.trace.base_tracer import BaseTracer


class OfflineJudgmentSpanProcessor(SimpleSpanProcessor):
    """Synchronous span processor used by ``OfflineTracer``.

    Each span is exported once, on end, via ``SimpleSpanProcessor``. Unlike
    ``JudgmentSpanProcessor`` this processor intentionally does *not*:

      - add ``judgment.update_id`` to span attributes,
      - support ``emit_partial`` (partial-span streaming), or
      - track per-span mutable state (counters, lists).

    Offline traces are short-lived, deterministic, and never re-emitted, so
    none of the live-monitoring machinery applies. Integrations that call
    ``emit_partial`` / ``state_*`` against an OfflineTracer get silent
    no-ops, satisfying ``JudgmentSpanProcessorLike`` so OfflineTracer can
    stand in wherever a normal Tracer is expected.

    On every *root* span end this processor appends a new ``Example`` to
    the caller-supplied ``dataset`` list, populated with the static
    ``example_fields`` plus an ``offline_trace_id`` referencing the trace.
    """

    __slots__ = (
        "tracer",
        "_dataset",
        "_example_fields",
        "_dataset_lock",
        "_seen_trace_ids",
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
        self._baggage_processor = JudgmentBaggageProcessor()

    # ------------------------------------------------------------------ #
    #  JudgmentSpanProcessorLike no-ops                                  #
    # ------------------------------------------------------------------ #

    def emit_partial(self) -> None:
        return None

    def state_set(self, span_context: SpanContext, key: str, value: Any) -> None:
        return None

    def state_get(
        self,
        span_context: SpanContext,
        key: str,
        default: Any = None,
    ) -> Any:
        return default

    def state_incr(self, span_context: SpanContext, key: str) -> int:
        return 0

    def state_append(self, span_context: SpanContext, key: str, item: Any) -> list[Any]:
        return [item]

    # ------------------------------------------------------------------ #
    #  Span lifecycle                                                    #
    # ------------------------------------------------------------------ #

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

    @dont_throw
    def on_end(self, span: ReadableSpan) -> None:
        self._maybe_create_example(span)
        super().on_end(span)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
