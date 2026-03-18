from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from together import Together, AsyncTogether


@pytest.fixture
def sync_together_client():
    return Together(api_key="test-key")


@pytest.fixture
def async_together_client():
    return AsyncTogether(api_key="test-key")


def make_together_response(
    model="meta-llama/Llama-3-8b-chat-hf",
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
