from abc import ABC, abstractmethod
from typing import List, Dict

from opentelemetry.sdk.trace import ReadableSpan


class ABCSpanStore(ABC):
    @abstractmethod
    def add(self, *spans: ReadableSpan): ...

    @abstractmethod
    def get_all(self) -> List[ReadableSpan]: ...

    @abstractmethod
    def get_by_trace_id(self, trace_id: str) -> List[ReadableSpan]: ...

    @abstractmethod
    def clear_trace(self, trace_id: str): ...


class SpanStore(ABCSpanStore):
    __slots__ = ("_spans_by_trace",)

    _spans_by_trace: Dict[str, List[ReadableSpan]]

    def __init__(self):
        self._spans_by_trace = {}

    def add(self, *spans: ReadableSpan):
        for span in spans:
            context = span.get_span_context()
            if context is None:
                continue
            # Convert trace_id to hex string per OTEL spec
            trace_id = format(context.trace_id, "032x")
            if trace_id not in self._spans_by_trace:
                self._spans_by_trace[trace_id] = []
            self._spans_by_trace[trace_id].append(span)

    def get_all(self) -> List[ReadableSpan]:
        all_spans = []
        for spans in self._spans_by_trace.values():
            all_spans.extend(spans)
        return all_spans

    def get_by_trace_id(self, trace_id: str) -> List[ReadableSpan]:
        """Get all spans for a specific trace ID (32-char hex string)."""
        return self._spans_by_trace.get(trace_id, [])

    def clear_trace(self, trace_id: str):
        """Clear all spans for a specific trace ID (32-char hex string)."""
        if trace_id in self._spans_by_trace:
            del self._spans_by_trace[trace_id]

    def __repr__(self) -> str:
        total_spans = sum(len(spans) for spans in self._spans_by_trace.values())
        return (
            f"SpanStore(traces={len(self._spans_by_trace)}, total_spans={total_spans})"
        )
