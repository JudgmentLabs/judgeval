from __future__ import annotations
from typing import Any, TypeAlias

from judgeval.tracer.llm.llm_openai.config import (
    HAS_OPENAI,
    openai_OpenAI,
    openai_AsyncOpenAI,
    openai_ChatCompletion,
    openai_Response,
    openai_ParsedChatCompletion,
)
from judgeval.tracer.llm.llm_together.config import (
    HAS_TOGETHER,
    together_Together,
    together_AsyncTogether,
)
from judgeval.tracer.llm.llm_anthropic.config import (
    HAS_ANTHROPIC,
    anthropic_Anthropic,
    anthropic_AsyncAnthropic,
)
from judgeval.tracer.llm.llm_google.config import (
    HAS_GOOGLE_GENAI,
    google_genai_Client,
    google_genai_AsyncClient,
)
from judgeval.tracer.llm.llm_groq.config import (
    HAS_GROQ,
    groq_Groq,
    groq_AsyncGroq,
)


# TODO: if we support dependency groups we can have this better type, but during runtime, we do
# not know which clients an end user might have installed.
ApiClient: TypeAlias = Any

__all__ = [
    "ApiClient",
    # OpenAI
    "HAS_OPENAI",
    "openai_OpenAI",
    "openai_AsyncOpenAI",
    "openai_ChatCompletion",
    "openai_Response",
    "openai_ParsedChatCompletion",
    # Together
    "HAS_TOGETHER",
    "together_Together",
    "together_AsyncTogether",
    # Anthropic
    "HAS_ANTHROPIC",
    "anthropic_Anthropic",
    "anthropic_AsyncAnthropic",
    # Google GenAI
    "HAS_GOOGLE_GENAI",
    "google_genai_Client",
    "google_genai_AsyncClient",
    # Groq
    "HAS_GROQ",
    "groq_Groq",
    "groq_AsyncGroq",
]
