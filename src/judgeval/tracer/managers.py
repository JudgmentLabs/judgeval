from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from judgeval.tracer import Tracer


@contextmanager
def sync_span_context(
    tracer: Tracer,
    name: str,
    span_attributes: Optional[Dict[str, str]] = None,
):
    if span_attributes is None:
        span_attributes = {}

    with tracer.get_tracer().start_as_current_span(
        name=name,
        attributes=span_attributes,
    ) as span:
        yield span


@asynccontextmanager
async def async_span_context(
    tracer: Tracer, name: str, span_attributes: Optional[Dict[str, str]] = None
):
    if span_attributes is None:
        span_attributes = {}

    with tracer.get_tracer().start_as_current_span(
        name=name,
        attributes=span_attributes,
    ) as span:
        yield span
