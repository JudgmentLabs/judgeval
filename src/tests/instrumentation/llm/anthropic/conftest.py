from __future__ import annotations

import pytest
from anthropic import Anthropic, AsyncAnthropic


@pytest.fixture
def sync_anthropic_client():
    return Anthropic(api_key="test-key")


@pytest.fixture
def async_anthropic_client():
    return AsyncAnthropic(api_key="test-key")


def make_message(
    model="claude-3-5-sonnet-latest", content="Hello", input_tokens=10, output_tokens=5
):
    from unittest.mock import MagicMock
    from anthropic.types import Message, TextBlock, Usage

    msg = MagicMock(spec=Message)
    msg.model = model
    msg.content = [MagicMock(spec=TextBlock, type="text", text=content)]
    msg.usage = MagicMock(
        spec=Usage,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )
    return msg
