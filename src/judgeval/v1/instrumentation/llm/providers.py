from __future__ import annotations
from typing import Any, TypeAlias

from judgeval.v1.instrumentation.llm.llm_openai.config import HAS_OPENAI
from judgeval.v1.instrumentation.llm.llm_together.config import HAS_TOGETHER
from judgeval.v1.instrumentation.llm.llm_anthropic.config import HAS_ANTHROPIC
from judgeval.v1.instrumentation.llm.llm_google.config import HAS_GOOGLE_GENAI

# TODO: if we support dependency groups we can have this better type, but during runtime, we do
# not know which clients an end user might have installed.
ApiClient: TypeAlias = Any

__all__ = [
    "ApiClient",
    "HAS_OPENAI",
    "HAS_TOGETHER",
    "HAS_ANTHROPIC",
    "HAS_GOOGLE_GENAI",
]
