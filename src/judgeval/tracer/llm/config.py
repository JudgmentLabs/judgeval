from __future__ import annotations
from typing import TYPE_CHECKING
from judgeval.logger import judgeval_logger

from judgeval.tracer.llm.constants import ProviderType
from judgeval.tracer.llm.providers import (
    HAS_OPENAI,
    HAS_TOGETHER,
    HAS_ANTHROPIC,
    HAS_GOOGLE_GENAI,
    HAS_GROQ,
    ApiClient,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer


def _detect_provider(client: ApiClient) -> ProviderType:
    if HAS_OPENAI:
        from judgeval.tracer.llm.providers import openai_OpenAI, openai_AsyncOpenAI

        assert openai_OpenAI is not None, "OpenAI client not found"
        assert openai_AsyncOpenAI is not None, "OpenAI async client not found"
        if isinstance(client, (openai_OpenAI, openai_AsyncOpenAI)):
            return ProviderType.OPENAI

    if HAS_ANTHROPIC:
        from judgeval.tracer.llm.providers import (
            anthropic_Anthropic,
            anthropic_AsyncAnthropic,
        )

        assert anthropic_Anthropic is not None, "Anthropic client not found"
        assert anthropic_AsyncAnthropic is not None, "Anthropic async client not found"
        if isinstance(client, (anthropic_Anthropic, anthropic_AsyncAnthropic)):
            return ProviderType.ANTHROPIC

    if HAS_TOGETHER:
        from judgeval.tracer.llm.providers import (
            together_Together,
            together_AsyncTogether,
        )

        assert together_Together is not None, "Together client not found"
        assert together_AsyncTogether is not None, "Together async client not found"
        if isinstance(client, (together_Together, together_AsyncTogether)):
            return ProviderType.TOGETHER

    if HAS_GOOGLE_GENAI:
        from judgeval.tracer.llm.providers import (
            google_genai_Client,
            google_genai_AsyncClient,
        )

        assert google_genai_Client is not None, "Google GenAI client not found"
        assert google_genai_AsyncClient is not None, (
            "Google GenAI async client not found"
        )
        if isinstance(client, (google_genai_Client, google_genai_AsyncClient)):
            return ProviderType.GOOGLE

    if HAS_GROQ:
        from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

        assert groq_Groq is not None, "Groq client not found"
        assert groq_AsyncGroq is not None, "Groq async client not found"
        if isinstance(client, (groq_Groq, groq_AsyncGroq)):
            return ProviderType.GROQ

    judgeval_logger.warning(
        f"Unknown client type {type(client)}, Trying to wrap as OpenAI-compatible. "
        "If this is a mistake or you think we should support this client, please file an issue at https://github.com/JudgmentLabs/judgeval/issues!"
    )

    return ProviderType.DEFAULT


def wrap_provider(tracer: Tracer, client: ApiClient) -> ApiClient:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, Google GenAI, and Groq clients.
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
    elif provider_type == ProviderType.GROQ:
        from .llm_groq.wrapper import wrap_groq_client

        return wrap_groq_client(tracer, client)
    else:
        # Default to OpenAI-compatible wrapping for unknown clients
        from .llm_openai.wrapper import wrap_openai_client

        return wrap_openai_client(tracer, client)
