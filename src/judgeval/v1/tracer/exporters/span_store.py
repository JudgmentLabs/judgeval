from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from opentelemetry.sdk.trace import ReadableSpan


class ABCSpanStore(ABC):
    @abstractmethod
    def add(self, *spans: ReadableSpan) -> None: ...

    @abstractmethod
    def get_all(self) -> List[ReadableSpan]: ...

    @abstractmethod
    def get_by_trace_id(self, trace_id: str) -> List[ReadableSpan]: ...

    @abstractmethod
    def clear_trace(self, trace_id: str) -> None: ...


class SpanStore(ABCSpanStore):
    __slots__ = ("_spans_by_trace",)

    def __init__(self) -> None:
        self._spans_by_trace: Dict[str, List[ReadableSpan]] = {}

    def add(self, *spans: ReadableSpan) -> None:
        for span in spans:
            context = span.get_span_context()
            if context is None:
                continue
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
        return self._spans_by_trace.get(trace_id, [])

    def clear_trace(self, trace_id: str) -> None:
        if trace_id in self._spans_by_trace:
            del self._spans_by_trace[trace_id]
