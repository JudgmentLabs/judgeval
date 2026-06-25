from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from openai import OpenAI, AsyncOpenAI

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


@pytest.fixture
def sync_minimax_client():
    return OpenAI(api_key="test-minimax-key", base_url=MINIMAX_BASE_URL)


@pytest.fixture
def async_minimax_client():
    return AsyncOpenAI(api_key="test-minimax-key", base_url=MINIMAX_BASE_URL)


def make_minimax_response(
    model="MiniMax-M2.7",
    content="Hello from MiniMax",
    prompt_tokens=10,
    completion_tokens=5,
):
    response = MagicMock()
    response.model = model
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return response
