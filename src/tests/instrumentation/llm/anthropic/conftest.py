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


def make_stream_chunks(input_tokens=10, start_output_tokens=5, final_output_tokens=42):
    """Return a list of fake RawMessageStreamEvent chunks for streaming tests.

    Produces three chunks in Anthropic's streaming protocol order:
    - message_start: carries the full Usage (input count + early output estimate)
    - content_block_delta: carries a text fragment
    - message_delta: carries MessageDeltaUsage with the authoritative output count
    """
    from unittest.mock import MagicMock
    from anthropic.types import Usage, MessageDeltaUsage

    start_chunk = MagicMock()
    start_chunk.type = "message_start"
    start_chunk.message = MagicMock()
    start_chunk.message.usage = MagicMock(
        spec=Usage,
        input_tokens=input_tokens,
        output_tokens=start_output_tokens,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )

    content_chunk = MagicMock()
    content_chunk.type = "content_block_delta"
    content_chunk.delta = MagicMock()
    content_chunk.delta.type = "text_delta"
    content_chunk.delta.text = "Hello"

    delta_chunk = MagicMock()
    delta_chunk.type = "message_delta"
    delta_chunk.usage = MagicMock(spec=MessageDeltaUsage, output_tokens=final_output_tokens)

    return [start_chunk, content_chunk, delta_chunk]
