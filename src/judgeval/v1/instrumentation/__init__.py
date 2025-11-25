from __future__ import annotations

from .llm import *
from .llm.providers import ApiClient


def wrap(client: ApiClient) -> ApiClient:
    from judgeval.v1.tracer.base_tracer import BaseTracer

    for tracer in BaseTracer._tracers:
        client = tracer.wrap(client)
    return client


__all__ = ["wrap_provider", "wrap"]
