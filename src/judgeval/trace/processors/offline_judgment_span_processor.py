from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from judgeval.trace.processors.judgment_span_processor import JudgmentSpanProcessor

if TYPE_CHECKING:
    from judgeval.data.example import Example
    from judgeval.trace.base_tracer import BaseTracer


class OfflineJudgmentSpanProcessor(JudgmentSpanProcessor):
    """Span processor used by ``OfflineTracer``.

    Extends ``JudgmentSpanProcessor`` (so it inherits batched export, span
    state, and partial-emit support) and additionally appends a new
    ``Example`` to the caller-supplied ``dataset`` list whenever a *root*
    span ends. Each emitted example carries the ``offline_trace_id`` of
    the trace plus any static ``example_fields`` configured at init time.
    """

    __slots__ = (
        "_dataset",
        "_example_fields",
        "_dataset_lock",
        "_seen_trace_ids",
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
        super().__init__(tracer, exporter)
        self._dataset = dataset
        self._example_fields: Dict[str, Any] = dict(example_fields or {})
        self._dataset_lock = threading.Lock()
        self._seen_trace_ids: set[str] = set()

    def _maybe_create_example(self, span: ReadableSpan) -> None:
        if span.parent is not None or not span.context:
            return

        trace_id_hex = format(span.context.trace_id, "032x")

        with self._dataset_lock:
            if trace_id_hex in self._seen_trace_ids:
                return
            self._seen_trace_ids.add(trace_id_hex)

        from judgeval.data.example import Example

        example = Example.create(
            **self._example_fields,
            offline_trace_id=trace_id_hex,
        )

        with self._dataset_lock:
            self._dataset.append(example)

    def on_end(self, span: ReadableSpan) -> None:
        try:
            self._maybe_create_example(span)
        finally:
            super().on_end(span)
