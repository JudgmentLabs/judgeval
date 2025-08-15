from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Dict, Optional
from judgeval.tracer.keys import AttributeKeys

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

    current_cost_context = tracer.get_current_cost_context()

    cost_context = {"cumulative_cost": 0.0}

    cost_token = current_cost_context.set(cost_context)

    try:
        with tracer.get_tracer().start_as_current_span(
            name=name,
            attributes=span_attributes,
        ) as span:
            # Set initial cumulative cost attribute
            span.set_attribute(AttributeKeys.JUDGMENT_CUMULATIVE_LLM_COST, 0.0)
            yield span
    finally:
        current_cost_context.reset(cost_token)
        child_cost = float(cost_context.get("cumulative_cost", 0.0))
        tracer.add_cost_to_current_context(child_cost)


@asynccontextmanager
async def async_span_context(
    tracer: Tracer, name: str, span_attributes: Optional[Dict[str, str]] = None
):
    if span_attributes is None:
        span_attributes = {}

    current_cost_context = tracer.get_current_cost_context()

    cost_context = {"cumulative_cost": 0.0}

    cost_token = current_cost_context.set(cost_context)

    try:
        with tracer.get_tracer().start_as_current_span(
            name=name,
            attributes=span_attributes,
        ) as span:
            span.set_attribute(AttributeKeys.JUDGMENT_CUMULATIVE_LLM_COST, 0.0)
            yield span
    finally:
        current_cost_context.reset(cost_token)
        child_cost = float(cost_context.get("cumulative_cost", 0.0))
        tracer.add_cost_to_current_context(child_cost)
