from __future__ import annotations

from typing import Callable, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from judgeval.trace.baggage import get_all

BaggageKeyPredicateT = Callable[[str], bool]
"""Predicate that decides which baggage keys are propagated to span attributes."""

ALLOW_ALL_BAGGAGE_KEYS: BaggageKeyPredicateT = lambda _: True  # noqa: E731
"""Default predicate that allows every baggage key."""


class JudgmentBaggageProcessor(SpanProcessor):
    """Copies OTel baggage entries onto span attributes at span start.

    When a span starts, this processor reads all baggage from the parent
    context and sets matching entries as span attributes. Use
    ``baggage_key_predicate`` to control which keys are propagated.

    Args:
        baggage_key_predicate: A callable that receives a baggage key and
            returns True if it should be copied to the span. Defaults to
            ``ALLOW_ALL_BAGGAGE_KEYS``.

    Examples:
        Allow only Judgment-prefixed baggage keys:

        ```python
        from judgeval.trace.processors import JudgmentBaggageProcessor

        processor = JudgmentBaggageProcessor(
            baggage_key_predicate=lambda k: k.startswith("judgment."),
        )
        ```
    """

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
