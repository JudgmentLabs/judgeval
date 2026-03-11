from __future__ import annotations

from typing import Callable, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from judgeval.v1.trace.baggage import get_all

BaggageKeyPredicateT = Callable[[str], bool]
ALLOW_ALL_BAGGAGE_KEYS: BaggageKeyPredicateT = lambda _: True  # noqa: E731


class JudgmentBaggageProcessor(SpanProcessor):
    def __init__(
        self,
        baggage_key_predicate: BaggageKeyPredicateT = ALLOW_ALL_BAGGAGE_KEYS,
    ) -> None:
        self._baggage_key_predicate = baggage_key_predicate

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        for key, value in get_all(parent_context).items():
            if self._baggage_key_predicate(key):
                span.set_attribute(key, value)  # type: ignore[arg-type]

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
