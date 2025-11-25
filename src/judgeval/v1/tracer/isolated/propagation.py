from __future__ import annotations

from typing import Optional

from opentelemetry import trace as trace_api
from opentelemetry.context.context import Context
from opentelemetry.trace import Span

from judgeval.v1.tracer.isolated.context import get_current


def set_span_in_context(span: Span, context: Optional[Context] = None) -> Context:
    if context is None:
        context = get_current()
    return trace_api.set_span_in_context(span, context)


def get_current_span(context: Optional[Context] = None) -> Span:
    if context is None:
        context = get_current()
    return trace_api.get_current_span(context)
