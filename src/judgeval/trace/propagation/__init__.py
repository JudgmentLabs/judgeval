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

from judgeval.trace.baggage.propagator import JudgmentBaggagePropagator

_global_textmap: TextMapPropagator = CompositePropagator(
    [TraceContextTextMapPropagator(), JudgmentBaggagePropagator()]
)


def get_global_textmap() -> TextMapPropagator:
    """Return the active composite propagator (W3C TraceContext + Judgment Baggage)."""
    return _global_textmap


def set_global_textmap(propagator: TextMapPropagator) -> None:
    """Replace the global text-map propagator."""
    global _global_textmap
    _global_textmap = propagator


def _resolve_context(context: Optional[Context]) -> Context:
    if context is not None:
        return context
    from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

    return JudgmentTracerProvider.get_instance().get_current_context()


def inject(
    carrier: object,
    context: Optional[Context] = None,
    setter: Setter = default_setter,
) -> None:
    """Inject trace context and baggage into an outgoing carrier (e.g. HTTP headers).

    Call this before making an outbound HTTP request to propagate the
    current trace across service boundaries.

    Args:
        carrier: A mutable mapping (typically a ``dict``) to write
            propagation headers into.
        context: The OTel context to inject. Defaults to the current
            Judgment context.
        setter: Strategy for writing values into the carrier.

    Examples:
        ```python
        headers = {}
        propagation.inject(headers)
        response = requests.get("https://api.example.com", headers=headers)
        ```
    """
    get_global_textmap().inject(
        carrier, context=_resolve_context(context), setter=setter
    )


def extract(
    carrier: object,
    context: Optional[Context] = None,
    getter: Getter = default_getter,
) -> Context:
    """Extract trace context and baggage from an incoming carrier.

    Call this when handling an inbound request to continue an existing
    trace started by an upstream service.

    Args:
        carrier: A mapping containing propagation headers
            (e.g. ``request.headers``).
        context: Base context to merge into. Defaults to the current
            Judgment context.
        getter: Strategy for reading values from the carrier.

    Returns:
        A new ``Context`` with the extracted trace and baggage data.

    Examples:
        ```python
        ctx = propagation.extract(request.headers)
        with Tracer.start_as_current_span("handle-request", context=ctx):
            ...
        ```
    """
    return get_global_textmap().extract(
        carrier, _resolve_context(context), getter=getter
    )


__all__ = [
    "get_global_textmap",
    "set_global_textmap",
    "inject",
    "extract",
]
