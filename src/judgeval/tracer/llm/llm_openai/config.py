from __future__ import annotations

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

__all__ = [
    "HAS_OPENAI",
    "openai_OpenAI",
    "openai_AsyncOpenAI",
    "openai_ChatCompletion",
    "openai_Response",
    "openai_ParsedChatCompletion",
]
