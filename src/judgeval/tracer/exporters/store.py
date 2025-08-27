from abc import ABC, abstractmethod
from typing import List

from opentelemetry.sdk.trace import ReadableSpan


class ABCSpanStore(ABC):
    @abstractmethod
    def add(self, *spans: ReadableSpan): ...

    @abstractmethod
    def get(self, id: str) -> ReadableSpan: ...

    @abstractmethod
    def get_all(self) -> List[ReadableSpan]: ...


class SpanStore(ABCSpanStore):
    __slots__ = ("spans",)

    spans: List[ReadableSpan]

    def __init__(self):
        self.spans = []

    def add(self, *spans: ReadableSpan):
        self.spans.extend(spans)

    def get(self, id: str) -> ReadableSpan:
        for span in self.spans:
            context = span.get_span_context()
            if context is None:
                continue
            if context.span_id == id:
                return span

        raise ValueError(f"Span with id {id} not found")

    def get_all(self) -> List[ReadableSpan]:
        return self.spans

    def __repr__(self) -> str:
        return f"SpanStore(spans={self.spans})"
