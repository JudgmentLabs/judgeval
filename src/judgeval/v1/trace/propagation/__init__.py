from __future__ import annotations

from typing import Optional

from opentelemetry.context import Context
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.textmap import (
    Getter,
    Setter,
    TextMapPropagator,
    default_getter,
    default_setter,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from judgeval.v1.trace.baggage.propagator import JudgmentBaggagePropagator

_global_textmap: TextMapPropagator = CompositePropagator(
    [TraceContextTextMapPropagator(), JudgmentBaggagePropagator()]
)


def get_global_textmap() -> TextMapPropagator:
    return _global_textmap


def set_global_textmap(propagator: TextMapPropagator) -> None:
    global _global_textmap
    _global_textmap = propagator


def _resolve_context(context: Optional[Context]) -> Context:
    if context is not None:
        return context
    from judgeval.v1.trace.judgment_tracer_provider import JudgmentTracerProvider

    return JudgmentTracerProvider.get_instance().get_current_context()


def inject(
    carrier: object,
    context: Optional[Context] = None,
    setter: Setter = default_setter,
) -> None:
    get_global_textmap().inject(
        carrier, context=_resolve_context(context), setter=setter
    )


def extract(
    carrier: object,
    context: Optional[Context] = None,
    getter: Getter = default_getter,
) -> Context:
    return get_global_textmap().extract(
        carrier, _resolve_context(context), getter=getter
    )


__all__ = [
    "get_global_textmap",
    "set_global_textmap",
    "inject",
    "extract",
]
