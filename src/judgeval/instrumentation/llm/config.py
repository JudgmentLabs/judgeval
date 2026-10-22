from __future__ import annotations
from typing import Any, TypeVar, cast
from judgeval.logger import judgeval_logger

from judgeval.instrumentation.llm.constants import ProviderType
from judgeval.instrumentation.llm.providers import (
    HAS_OPENAI,
    HAS_TOGETHER,
    HAS_ANTHROPIC,
    HAS_GOOGLE_GENAI,
    ApiClient,
)

T = TypeVar("T", bound=ApiClient)

ORCAROUTER_BASE_URL_MARKER = "api.orcarouter.ai"
ORCAROUTER_API_KEY_PREFIX = "sk-orca-"


def _is_orcarouter_client(client: Any) -> bool:
    """Return True when an OpenAI-compatible client is pointed at OrcaRouter.

    OrcaRouter is an OpenAI-compatible gateway (https://api.orcarouter.ai/v1).
    We detect it two ways so existing OrcaRouter users are picked up even
    before they switch base_urls:
      * the configured ``base_url`` contains ``api.orcarouter.ai``, or
      * the API key uses the ``sk-orca-`` prefix issued at
        https://www.orcarouter.ai.
    """
    base_url = getattr(client, "base_url", None)
    if base_url is not None and ORCAROUTER_BASE_URL_MARKER in str(base_url):
        return True

    api_key = getattr(client, "api_key", None)
    if isinstance(api_key, str) and api_key.startswith(ORCAROUTER_API_KEY_PREFIX):
        return True

    return False


def _detect_provider(client: ApiClient) -> ProviderType:
    if HAS_OPENAI:
        from openai import OpenAI, AsyncOpenAI

        if isinstance(client, (OpenAI, AsyncOpenAI)):
            if _is_orcarouter_client(client):
                return ProviderType.ORCAROUTER
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


def wrap_provider(client: T) -> T:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, Google GenAI, and OrcaRouter clients.
    Uses the active tracer via JudgmentTracerProvider.
    """
    provider_type = _detect_provider(client)

    if provider_type == ProviderType.OPENAI:
        from .llm_openai.wrapper import wrap_openai_client

        return cast(T, wrap_openai_client(client))
    elif provider_type == ProviderType.ANTHROPIC:
        from .llm_anthropic.wrapper import wrap_anthropic_client

        return cast(T, wrap_anthropic_client(client))
    elif provider_type == ProviderType.TOGETHER:
        from .llm_together.wrapper import wrap_together_client

        return cast(T, wrap_together_client(client))
    elif provider_type == ProviderType.GOOGLE:
        from .llm_google.wrapper import wrap_google_client

        return cast(T, wrap_google_client(client))
    elif provider_type == ProviderType.ORCAROUTER:
        from .llm_orcarouter.wrapper import wrap_orcarouter_client

        return cast(T, wrap_orcarouter_client(client))
    else:
        from .llm_openai.wrapper import wrap_openai_client

        return cast(T, wrap_openai_client(client))
