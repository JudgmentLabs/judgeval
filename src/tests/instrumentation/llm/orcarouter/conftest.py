from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from openai import OpenAI, AsyncOpenAI


@pytest.fixture
def sync_orcarouter_client():
    return OpenAI(
        api_key="sk-orca-test-key",
        base_url="https://api.orcarouter.ai/v1",
    )


@pytest.fixture
def async_orcarouter_client():
    return AsyncOpenAI(
        api_key="sk-orca-test-key",
        base_url="https://api.orcarouter.ai/v1",
    )


def make_orcarouter_response(
    model="orcarouter/auto",
    content="Hello",
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
