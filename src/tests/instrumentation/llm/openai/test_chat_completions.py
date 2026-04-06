from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.instrumentation.llm.llm_openai.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)
from tests.instrumentation.llm.openai.conftest import (
    make_chat_completion,
    make_chunk,
)


class TestSyncNonStreaming:
    def test_creates_span(self, tracer, collecting_exporter, sync_openai_client):
        completion = make_chat_completion()
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        assert any(s.name == "OPENAI_API_CALL" for s in collecting_exporter.spans)

    def test_span_has_llm_kind(self, tracer, collecting_exporter, sync_openai_client):
        completion = make_chat_completion()
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    def test_records_model_name(self, tracer, collecting_exporter, sync_openai_client):
        completion = make_chat_completion(model="gpt-4-turbo")
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4-turbo", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME) == "gpt-4-turbo"
        )

    def test_records_token_usage(self, tracer, collecting_exporter, sync_openai_client):
        completion = make_chat_completion(prompt_tokens=15, completion_tokens=8)
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 15
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 8

    def test_error_sets_span_error_status(
        self, tracer, collecting_exporter, sync_openai_client
    ):
        sync_openai_client.chat.completions.create = MagicMock(
            side_effect=RuntimeError("fail")
        )
        wrap_chat_completions_create_sync(sync_openai_client)
        with pytest.raises(RuntimeError):
            sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert span.status.status_code.name == "ERROR"

    def test_returns_result_unchanged(self, tracer, sync_openai_client):
        completion = make_chat_completion()
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        result = sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        assert result is completion

    def test_stainless_raw_response_bypasses_wrap(self, tracer, sync_openai_client):
        completion = make_chat_completion()
        original = MagicMock(return_value=completion)
        sync_openai_client.chat.completions.create = original
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(
            model="gpt-4",
            messages=[],
            extra_headers={"X-Stainless-Raw-Response": "stream"},
        )
        original.assert_called_once()


class TestSyncStreaming:
    def test_streaming_creates_span(
        self, tracer, collecting_exporter, sync_openai_client
    ):
        chunks = [make_chunk(content="Hi"), make_chunk(content=" there")]
        sync_openai_client.chat.completions.create = MagicMock(
            return_value=iter(chunks)
        )
        wrap_chat_completions_create_sync(sync_openai_client)
        stream = sync_openai_client.chat.completions.create(
            model="gpt-4", messages=[], stream=True
        )
        list(stream)
        assert any(s.name == "OPENAI_API_CALL" for s in collecting_exporter.spans)

    def test_streaming_accumulates_content(
        self, tracer, collecting_exporter, sync_openai_client
    ):
        chunks = [make_chunk(content="Hello"), make_chunk(content=" world")]
        sync_openai_client.chat.completions.create = MagicMock(
            return_value=iter(chunks)
        )
        wrap_chat_completions_create_sync(sync_openai_client)
        stream = sync_openai_client.chat.completions.create(
            model="gpt-4", messages=[], stream=True
        )
        list(stream)
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert "Hello" in (span.attributes.get(AttributeKeys.GEN_AI_COMPLETION) or "")


class TestAsyncNonStreaming:
    @pytest.mark.asyncio
    async def test_creates_span(self, tracer, collecting_exporter, async_openai_client):
        completion = make_chat_completion()
        async_openai_client.chat.completions.create = AsyncMock(return_value=completion)
        wrap_chat_completions_create_async(async_openai_client)
        await async_openai_client.chat.completions.create(model="gpt-4", messages=[])
        assert any(s.name == "OPENAI_API_CALL" for s in collecting_exporter.spans)

    @pytest.mark.asyncio
    async def test_records_model_name(
        self, tracer, collecting_exporter, async_openai_client
    ):
        completion = make_chat_completion(model="gpt-4")
        async_openai_client.chat.completions.create = AsyncMock(return_value=completion)
        wrap_chat_completions_create_async(async_openai_client)
        await async_openai_client.chat.completions.create(model="gpt-4", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert span.attributes.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME) == "gpt-4"

    @pytest.mark.asyncio
    async def test_error_sets_error_status(
        self, tracer, collecting_exporter, async_openai_client
    ):
        async_openai_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("fail")
        )
        wrap_chat_completions_create_async(async_openai_client)
        with pytest.raises(RuntimeError):
            await async_openai_client.chat.completions.create(
                model="gpt-4", messages=[]
            )
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert span.status.status_code.name == "ERROR"


class TestWrappingReplaceMethod:
    def test_wrap_replaces_create_method(self, tracer, sync_openai_client):
        original = sync_openai_client.chat.completions.create
        wrap_chat_completions_create_sync(sync_openai_client)
        assert sync_openai_client.chat.completions.create is not original

    def test_wrap_single_span_per_call(
        self, tracer, collecting_exporter, sync_openai_client
    ):
        completion = make_chat_completion()
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        api_spans = [
            s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL"
        ]
        assert len(api_spans) == 2
