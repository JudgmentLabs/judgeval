from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.instrumentation.llm.llm_together.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)
from tests.instrumentation.llm.together.conftest import make_together_response


class TestSyncNonStreaming:
    def test_creates_span(self, tracer, collecting_exporter, sync_together_client):
        response = make_together_response()
        sync_together_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_together_client)
        sync_together_client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf", messages=[]
        )
        assert any(s.name == "TOGETHER_API_CALL" for s in collecting_exporter.spans)

    def test_span_has_llm_kind(self, tracer, collecting_exporter, sync_together_client):
        response = make_together_response()
        sync_together_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_together_client)
        sync_together_client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "TOGETHER_API_CALL"
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    def test_records_token_usage(
        self, tracer, collecting_exporter, sync_together_client
    ):
        response = make_together_response(prompt_tokens=15, completion_tokens=8)
        sync_together_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_together_client)
        sync_together_client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "TOGETHER_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 15
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 8

    def test_error_sets_error_status(
        self, tracer, collecting_exporter, sync_together_client
    ):
        sync_together_client.chat.completions.create = MagicMock(
            side_effect=RuntimeError("fail")
        )
        wrap_chat_completions_create_sync(sync_together_client)
        with pytest.raises(RuntimeError):
            sync_together_client.chat.completions.create(
                model="meta-llama/Llama-3-8b-chat-hf", messages=[]
            )
        span = next(
            s for s in collecting_exporter.spans if s.name == "TOGETHER_API_CALL"
        )
        assert span.status.status_code.name == "ERROR"

    def test_returns_result(self, tracer, sync_together_client):
        response = make_together_response()
        sync_together_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_together_client)
        result = sync_together_client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf", messages=[]
        )
        assert result is response


class TestAsyncNonStreaming:
    @pytest.mark.asyncio
    async def test_creates_span(
        self, tracer, collecting_exporter, async_together_client
    ):
        response = make_together_response()
        async_together_client.chat.completions.create = AsyncMock(return_value=response)
        wrap_chat_completions_create_async(async_together_client)
        await async_together_client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf", messages=[]
        )
        assert any(s.name == "TOGETHER_API_CALL" for s in collecting_exporter.spans)
