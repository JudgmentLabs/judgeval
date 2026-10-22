from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.instrumentation.llm.llm_google.generate_content import (
    wrap_generate_content_stream_async,
)


def make_google_stream_chunk(text=None, prompt_tokens=None, completion_tokens=None):
    chunk = MagicMock()
    chunk.text = text
    chunk.model_version = None
    if prompt_tokens is not None or completion_tokens is not None:
        chunk.usage_metadata = MagicMock(
            prompt_token_count=prompt_tokens or 0,
            candidates_token_count=completion_tokens or 0,
            cached_content_token_count=None,
        )
    else:
        chunk.usage_metadata = None
    return chunk


def make_async_stream(*chunks):
    async def _gen(**kwargs):
        for chunk in chunks:
            yield chunk

    return _gen


class TestGoogleGenerateContentStreamAsync:
    @pytest.mark.asyncio
    async def test_creates_span(self, tracer, collecting_exporter, google_client):
        chunk = make_google_stream_chunk(text="Hi")
        google_client.aio.models.generate_content_stream = make_async_stream(chunk)
        wrap_generate_content_stream_async(google_client)
        async for _ in google_client.aio.models.generate_content_stream(
            model="gemini-2.0-flash", contents="hello"
        ):
            pass
        assert any(s.name == "GOOGLE_API_CALL" for s in collecting_exporter.spans)

    @pytest.mark.asyncio
    async def test_span_has_llm_kind(
        self, tracer, collecting_exporter, google_client
    ):
        chunk = make_google_stream_chunk(text="Hi")
        google_client.aio.models.generate_content_stream = make_async_stream(chunk)
        wrap_generate_content_stream_async(google_client)
        async for _ in google_client.aio.models.generate_content_stream(
            model="gemini-2.0-flash", contents="hello"
        ):
            pass
        span = next(
            s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL"
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    @pytest.mark.asyncio
    async def test_accumulates_text_across_chunks(
        self, tracer, collecting_exporter, google_client
    ):
        google_client.aio.models.generate_content_stream = make_async_stream(
            make_google_stream_chunk(text="Hello "),
            make_google_stream_chunk(text="world"),
            make_google_stream_chunk(text=None, prompt_tokens=8, completion_tokens=2),
        )
        wrap_generate_content_stream_async(google_client)
        async for _ in google_client.aio.models.generate_content_stream(
            model="gemini-2.0-flash", contents="hello"
        ):
            pass
        span = next(
            s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL"
        )
        assert span.attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Hello world"

    @pytest.mark.asyncio
    async def test_records_token_usage_from_final_chunk(
        self, tracer, collecting_exporter, google_client
    ):
        google_client.aio.models.generate_content_stream = make_async_stream(
            make_google_stream_chunk(text="A"),
            make_google_stream_chunk(text="B", prompt_tokens=15, completion_tokens=7),
        )
        wrap_generate_content_stream_async(google_client)
        async for _ in google_client.aio.models.generate_content_stream(
            model="gemini-2.0-flash", contents="hello"
        ):
            pass
        span = next(
            s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 15
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 7

    @pytest.mark.asyncio
    async def test_chunks_without_text_not_accumulated(
        self, tracer, collecting_exporter, google_client
    ):
        google_client.aio.models.generate_content_stream = make_async_stream(
            make_google_stream_chunk(text=None),
            make_google_stream_chunk(text="Only this"),
        )
        wrap_generate_content_stream_async(google_client)
        async for _ in google_client.aio.models.generate_content_stream(
            model="gemini-2.0-flash", contents="hello"
        ):
            pass
        span = next(
            s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL"
        )
        assert span.attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Only this"

    @pytest.mark.asyncio
    async def test_error_sets_error_status(
        self, tracer, collecting_exporter, google_client
    ):
        async def failing_stream(**kwargs):
            yield make_google_stream_chunk(text="A")
            raise RuntimeError("stream failed")

        google_client.aio.models.generate_content_stream = failing_stream
        wrap_generate_content_stream_async(google_client)
        with pytest.raises(RuntimeError):
            async for _ in google_client.aio.models.generate_content_stream(
                model="gemini-2.0-flash", contents="hello"
            ):
                pass
        span = next(
            s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL"
        )
        assert span.status.status_code.name == "ERROR"

    @pytest.mark.asyncio
    async def test_wrap_replaces_method(self, tracer, google_client):
        original = google_client.aio.models.generate_content_stream
        google_client.aio.models.generate_content_stream = make_async_stream(
            make_google_stream_chunk(text="hi")
        )
        wrap_generate_content_stream_async(google_client)
        assert google_client.aio.models.generate_content_stream is not original

    @pytest.mark.asyncio
    async def test_yields_all_chunks_unchanged(self, tracer, google_client):
        chunk1 = make_google_stream_chunk(text="A")
        chunk2 = make_google_stream_chunk(text="B")
        google_client.aio.models.generate_content_stream = make_async_stream(
            chunk1, chunk2
        )
        wrap_generate_content_stream_async(google_client)
        result = [
            chunk
            async for chunk in google_client.aio.models.generate_content_stream(
                model="gemini-2.0-flash", contents="hello"
            )
        ]
        assert result == [chunk1, chunk2]
