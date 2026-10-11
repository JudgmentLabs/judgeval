from __future__ import annotations

import asyncio
import json
import threading

import httpx
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


def _sse_chunk(delta, finish_reason=None, usage=None):
    choices = (
        []
        if delta is None
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    )
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4",
        "choices": choices,
        "usage": usage,
    }


def _sse_handler(request: httpx.Request) -> httpx.Response:
    """Serve a real chat completion SSE stream ("Hello"), with a usage chunk
    when the request asks for one."""
    body = json.loads(request.content)
    events = [
        _sse_chunk({"role": "assistant", "content": "Hel"}),
        _sse_chunk({"content": "lo"}),
        _sse_chunk({}, finish_reason="stop"),
    ]
    if (body.get("stream_options") or {}).get("include_usage"):
        usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        events.append(_sse_chunk(None, usage=usage))
    payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=(payload + "data: [DONE]\n\n").encode(),
    )


@pytest.fixture
def sse_sync_openai_client():
    return OpenAI(
        api_key="test-key",
        base_url="http://localhost",
        http_client=httpx.Client(transport=httpx.MockTransport(_sse_handler)),
    )


@pytest.fixture
def sse_async_openai_client():
    return AsyncOpenAI(
        api_key="test-key",
        base_url="http://localhost",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_sse_handler)),
    )


_GATED_EVENTS = [
    _sse_chunk({"role": "assistant", "content": "Hel"}),
    _sse_chunk({"content": "lo"}),
    _sse_chunk({}, finish_reason="stop"),
]


def _sse_bytes(event):
    return f"data: {json.dumps(event)}\n\n".encode()


class GatedSSE(httpx.SyncByteStream):
    """Sends the first chunk, then blocks until ``release`` is set and stops
    early if the response was closed in the meantime."""

    def __init__(self):
        self.reading = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def __iter__(self):
        yield _sse_bytes(_GATED_EVENTS[0])
        self.reading.set()
        self.release.wait(timeout=5)
        for event in _GATED_EVENTS[1:]:
            if self.closed:
                return
            yield _sse_bytes(event)
        yield b"data: [DONE]\n\n"

    def close(self):
        self.closed = True


class AsyncGatedSSE(httpx.AsyncByteStream):
    """Async counterpart of ``GatedSSE``."""

    def __init__(self):
        self.reading = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield _sse_bytes(_GATED_EVENTS[0])
        self.reading.set()
        await asyncio.wait_for(self.release.wait(), timeout=5)
        for event in _GATED_EVENTS[1:]:
            if self.closed:
                return
            yield _sse_bytes(event)
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


@pytest.fixture
def gated_sync_openai_client():
    body = GatedSSE()

    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body
        )

    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, body


@pytest.fixture
def gated_async_openai_client():
    body = AsyncGatedSSE()

    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="http://localhost",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client, body
