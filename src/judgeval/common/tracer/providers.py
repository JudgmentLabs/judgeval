from __future__ import annotations
import logging
from typing import Any, TypeAlias


logger = logging.getLogger(__name__)
# TODO: Have functions that assert and return the relevant exports when the client is installed.
# The method should raise if the user tries to access client information that doesnt exist.

HAS_OPENAI = False
openai_OpenAI = None
openai_AsyncOpenAI = None
openai_ChatCompletion = None
openai_Response = None
openai_ParsedChatCompletion = None

try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.responses.response import Response
    from openai.types.chat import ParsedChatCompletion

    openai_OpenAI = OpenAI
    openai_AsyncOpenAI = AsyncOpenAI
    openai_ChatCompletion = ChatCompletion
    openai_Response = Response
    openai_ParsedChatCompletion = ParsedChatCompletion
    HAS_OPENAI = True
except ImportError:
    pass


HAS_TOGETHER = False
together_Together = None
together_AsyncTogether = None

try:
    from together import Together, AsyncTogether

    together_Together = Together
    together_AsyncTogether = AsyncTogether
    HAS_TOGETHER = True
except ImportError:
    pass


HAS_ANTHROPIC = False
anthropic_Anthropic = None
anthropic_AsyncAnthropic = None

try:
    from anthropic import Anthropic, AsyncAnthropic

    anthropic_Anthropic = Anthropic
    anthropic_AsyncAnthropic = AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass


HAS_GOOGLE_GENAI = False
google_genai_Client = None
google_genai_cleint_AsyncClient = None

try:
    from google.genai import Client
    from google.genai.client import AsyncClient

    google_genai_Client = Client
    google_genai_AsyncClient = AsyncClient
    HAS_GOOGLE_GENAI = True
except ImportError:
    pass


HAS_GROQ = False
groq_Groq = None
groq_AsyncGroq = None

try:
    from groq import Groq, AsyncGroq

    groq_Groq = Groq
    groq_AsyncGroq = AsyncGroq
    HAS_GROQ = True
except ImportError:
    pass


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
