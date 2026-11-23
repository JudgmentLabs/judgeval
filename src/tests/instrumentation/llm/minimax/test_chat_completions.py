from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.instrumentation.llm.llm_minimax.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)
from judgeval.instrumentation.llm.llm_minimax.config import is_minimax_client
from judgeval.instrumentation.llm.config import _detect_provider
from judgeval.instrumentation.llm.constants import ProviderType
from tests.instrumentation.llm.minimax.conftest import make_minimax_response


class TestIsMinimaxClient:
    def test_openai_client_with_minimax_base_url(self):
        from openai import OpenAI

        client = OpenAI(api_key="test", base_url="https://api.minimax.io/v1")
        assert is_minimax_client(client) is True

    def test_openai_client_without_minimax_base_url(self):
        from openai import OpenAI

        client = OpenAI(api_key="test", base_url="https://api.openai.com/v1")
        assert is_minimax_client(client) is False

    def test_non_client_object(self):
        assert is_minimax_client(object()) is False


class TestDetectProvider:
    def test_detects_minimax_from_openai_client(self):
        from openai import OpenAI

        client = OpenAI(api_key="test", base_url="https://api.minimax.io/v1")
        result = _detect_provider(client)
        assert result == ProviderType.MINIMAX

    def test_regular_openai_client_is_not_minimax(self):
        from openai import OpenAI

        client = OpenAI(api_key="test", base_url="https://api.openai.com/v1")
        result = _detect_provider(client)
        assert result == ProviderType.OPENAI


class TestSyncNonStreaming:
    def test_creates_span(self, tracer, collecting_exporter, sync_minimax_client):
        response = make_minimax_response()
        sync_minimax_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_minimax_client)
        sync_minimax_client.chat.completions.create(
            model="MiniMax-M2.7", messages=[]
        )
        assert any(s.name == "MINIMAX_API_CALL" for s in collecting_exporter.spans)

    def test_span_has_llm_kind(self, tracer, collecting_exporter, sync_minimax_client):
        response = make_minimax_response()
        sync_minimax_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_minimax_client)
        sync_minimax_client.chat.completions.create(
            model="MiniMax-M2.7", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "MINIMAX_API_CALL"
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    def test_model_name_has_minimax_prefix(
        self, tracer, collecting_exporter, sync_minimax_client
    ):
        response = make_minimax_response()
        sync_minimax_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_minimax_client)
        sync_minimax_client.chat.completions.create(
            model="MiniMax-M2.7", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "MINIMAX_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME)
            == "minimax/MiniMax-M2.7"
        )

    def test_records_token_usage(
        self, tracer, collecting_exporter, sync_minimax_client
    ):
        response = make_minimax_response(prompt_tokens=20, completion_tokens=10)
        sync_minimax_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_minimax_client)
        sync_minimax_client.chat.completions.create(
            model="MiniMax-M2.7", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "MINIMAX_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 20
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 10

    def test_error_sets_error_status(
        self, tracer, collecting_exporter, sync_minimax_client
    ):
        sync_minimax_client.chat.completions.create = MagicMock(
            side_effect=RuntimeError("api error")
        )
        wrap_chat_completions_create_sync(sync_minimax_client)
        with pytest.raises(RuntimeError):
            sync_minimax_client.chat.completions.create(
                model="MiniMax-M2.7", messages=[]
            )
        span = next(
            s for s in collecting_exporter.spans if s.name == "MINIMAX_API_CALL"
        )
        assert span.status.status_code.name == "ERROR"

    def test_returns_result(self, tracer, sync_minimax_client):
        response = make_minimax_response()
        sync_minimax_client.chat.completions.create = MagicMock(return_value=response)
        wrap_chat_completions_create_sync(sync_minimax_client)
        result = sync_minimax_client.chat.completions.create(
            model="MiniMax-M2.7", messages=[]
        )
        assert result is response


class TestAsyncNonStreaming:
    @pytest.mark.asyncio
    async def test_creates_span(
        self, tracer, collecting_exporter, async_minimax_client
    ):
        response = make_minimax_response()
        async_minimax_client.chat.completions.create = AsyncMock(return_value=response)
        wrap_chat_completions_create_async(async_minimax_client)
        await async_minimax_client.chat.completions.create(
            model="MiniMax-M2.7", messages=[]
        )
        assert any(s.name == "MINIMAX_API_CALL" for s in collecting_exporter.spans)

    @pytest.mark.asyncio
    async def test_model_name_has_minimax_prefix(
        self, tracer, collecting_exporter, async_minimax_client
    ):
        response = make_minimax_response()
        async_minimax_client.chat.completions.create = AsyncMock(return_value=response)
        wrap_chat_completions_create_async(async_minimax_client)
        await async_minimax_client.chat.completions.create(
            model="MiniMax-M2.7-highspeed", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "MINIMAX_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME)
            == "minimax/MiniMax-M2.7-highspeed"
        )

    @pytest.mark.asyncio
    async def test_records_token_usage(
        self, tracer, collecting_exporter, async_minimax_client
    ):
        response = make_minimax_response(prompt_tokens=15, completion_tokens=8)
        async_minimax_client.chat.completions.create = AsyncMock(return_value=response)
        wrap_chat_completions_create_async(async_minimax_client)
        await async_minimax_client.chat.completions.create(
            model="MiniMax-M2.7", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "MINIMAX_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 15
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 8
