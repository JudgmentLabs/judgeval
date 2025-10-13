from __future__ import annotations
from typing import TYPE_CHECKING
from judgeval.logger import judgeval_logger

from judgeval.tracer.llm.constants import ProviderType
from judgeval.tracer.llm.providers import (
    HAS_OPENAI,
    HAS_TOGETHER,
    HAS_ANTHROPIC,
    HAS_GOOGLE_GENAI,
    ApiClient,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer


def _detect_provider(client: ApiClient) -> ProviderType:
    if HAS_OPENAI:
        from openai import OpenAI, AsyncOpenAI

        if isinstance(client, (OpenAI, AsyncOpenAI)):
            return ProviderType.OPENAI

    if HAS_ANTHROPIC:
        from anthropic import Anthropic, AsyncAnthropic

        if isinstance(client, (Anthropic, AsyncAnthropic)):
            return ProviderType.ANTHROPIC

    if HAS_TOGETHER:
        from together import Together, AsyncTogether  # type: ignore[import-untyped]

        if isinstance(client, (Together, AsyncTogether)):
            return ProviderType.TOGETHER

    if HAS_GOOGLE_GENAI:
        from google.genai import Client as GoogleClient

        if isinstance(client, GoogleClient):
            return ProviderType.GOOGLE

    judgeval_logger.warning(
        f"Unknown client type {type(client)}, Trying to wrap as OpenAI-compatible. "
        "If this is a mistake or you think we should support this client, please file an issue at https://github.com/JudgmentLabs/judgeval/issues!"
    )

    return ProviderType.DEFAULT


def wrap_provider(tracer: Tracer, client: ApiClient) -> ApiClient:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, and Google GenAI clients.
    """
    provider_type = _detect_provider(client)

    if provider_type == ProviderType.OPENAI:
        from .llm_openai.wrapper import wrap_openai_client

        return wrap_openai_client(tracer, client)
    elif provider_type == ProviderType.ANTHROPIC:
        from .llm_anthropic.wrapper import wrap_anthropic_client

        return wrap_anthropic_client(tracer, client)
    elif provider_type == ProviderType.TOGETHER:
        from .llm_together.wrapper import wrap_together_client

        return wrap_together_client(tracer, client)
    elif provider_type == ProviderType.GOOGLE:
        from .llm_google.wrapper import wrap_google_client

        return wrap_google_client(tracer, client)
    else:
        # Default to OpenAI-compatible wrapping for unknown clients
        from .llm_openai.wrapper import wrap_openai_client

        return wrap_openai_client(tracer, client)
