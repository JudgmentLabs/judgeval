from __future__ import annotations
from typing import TypeVar

from .llm import *
from .llm.providers import ApiClient
from .llm.config import wrap_provider

T = TypeVar("T", bound=ApiClient)


def wrap(client: T) -> T:
    """Wrap an API client to add tracing capabilities.
    Uses the active tracer via ProxyTracerProvider."""
    return wrap_provider(client)


__all__ = ["wrap_provider", "wrap"]
