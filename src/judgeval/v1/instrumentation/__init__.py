from __future__ import annotations
from typing import TypeVar

from .llm import *
from .llm.providers import ApiClient
from .llm.config import wrap_provider

T = TypeVar("T", bound=ApiClient)


def wrap(client: T) -> T:
    """Wrap a supported LLM client to add automatic tracing.

    Supports OpenAI, Anthropic, Together, and Google GenAI clients.
    Uses the active tracer via ``JudgmentTracerProvider``.

    Args:
        client: An LLM provider client instance to wrap.

    Returns:
        The same client instance, patched with tracing instrumentation.
    """
    return wrap_provider(client)


__all__ = ["wrap_provider", "wrap"]
