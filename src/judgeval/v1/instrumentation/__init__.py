from __future__ import annotations
from typing import TypeVar

from .llm import *
from .llm.providers import ApiClient

T = TypeVar("T", bound=ApiClient)


def wrap(client: T) -> T:
    from judgeval.v1.tracer.base_tracer import BaseTracer

    for tracer in BaseTracer._tracers:
        client = tracer.wrap(client)
    return client


__all__ = ["wrap_provider", "wrap"]
