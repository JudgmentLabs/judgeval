from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage


def make_chat_completion(
    model="gpt-4", content="Hello", prompt_tokens=10, completion_tokens=5
):
    return MagicMock(
        spec=ChatCompletion,
        model=model,
        choices=[MagicMock(message=MagicMock(content=content))],
        usage=MagicMock(
            spec=CompletionUsage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=None,
        ),
    )


def make_chunk(content=None, usage=None):
    chunk = MagicMock(spec=ChatCompletionChunk)
    if content is not None:
        chunk.choices = [MagicMock(delta=MagicMock(content=content))]
    else:
        chunk.choices = []
    chunk.usage = usage
    return chunk


@pytest.fixture
def sync_openai_client():
    return OpenAI(api_key="test-key", base_url="http://localhost")


@pytest.fixture
def async_openai_client():
    return AsyncOpenAI(api_key="test-key", base_url="http://localhost")
