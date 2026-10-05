from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.instrumentation.llm.llm_orcarouter.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)
from tests.instrumentation.llm.orcarouter.conftest import make_orcarouter_response


class TestSyncNonStreaming:
    def test_creates_span(self, tracer, collecting_exporter, sync_orcarouter_client):
        response = make_orcarouter_response()
        sync_orcarouter_client.chat.completions.create = MagicMock(
            return_value=response
        )
        wrap_chat_completions_create_sync(sync_orcarouter_client)
        sync_orcarouter_client.chat.completions.create(
            model="orcarouter/auto", messages=[]
        )
        assert any(s.name == "ORCAROUTER_API_CALL" for s in collecting_exporter.spans)

    def test_span_has_llm_kind(
        self, tracer, collecting_exporter, sync_orcarouter_client
    ):
        response = make_orcarouter_response()
        sync_orcarouter_client.chat.completions.create = MagicMock(
            return_value=response
        )
        wrap_chat_completions_create_sync(sync_orcarouter_client)
        sync_orcarouter_client.chat.completions.create(
            model="orcarouter/auto", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ORCAROUTER_API_CALL"
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    def test_records_model_name(
        self, tracer, collecting_exporter, sync_orcarouter_client
    ):
        response = make_orcarouter_response(model="orcarouter/fusion")
        sync_orcarouter_client.chat.completions.create = MagicMock(
            return_value=response
        )
        wrap_chat_completions_create_sync(sync_orcarouter_client)
        sync_orcarouter_client.chat.completions.create(
            model="orcarouter/fusion", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ORCAROUTER_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME)
            == "orcarouter/fusion"
        )

    def test_records_token_usage(
        self, tracer, collecting_exporter, sync_orcarouter_client
    ):
        response = make_orcarouter_response(prompt_tokens=15, completion_tokens=8)
        sync_orcarouter_client.chat.completions.create = MagicMock(
            return_value=response
        )
        wrap_chat_completions_create_sync(sync_orcarouter_client)
        sync_orcarouter_client.chat.completions.create(
            model="orcarouter/auto", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ORCAROUTER_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 15
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 8

    def test_error_sets_error_status(
        self, tracer, collecting_exporter, sync_orcarouter_client
    ):
        sync_orcarouter_client.chat.completions.create = MagicMock(
            side_effect=RuntimeError("fail")
        )
        wrap_chat_completions_create_sync(sync_orcarouter_client)
        with pytest.raises(RuntimeError):
            sync_orcarouter_client.chat.completions.create(
                model="orcarouter/auto", messages=[]
            )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ORCAROUTER_API_CALL"
        )
        assert span.status.status_code.name == "ERROR"

    def test_returns_result(self, tracer, sync_orcarouter_client):
        response = make_orcarouter_response()
        sync_orcarouter_client.chat.completions.create = MagicMock(
            return_value=response
        )
        wrap_chat_completions_create_sync(sync_orcarouter_client)
        result = sync_orcarouter_client.chat.completions.create(
            model="orcarouter/auto", messages=[]
        )
        assert result is response


class TestAsyncNonStreaming:
    @pytest.mark.asyncio
    async def test_creates_span(
        self, tracer, collecting_exporter, async_orcarouter_client
    ):
        response = make_orcarouter_response()
        async_orcarouter_client.chat.completions.create = AsyncMock(
            return_value=response
        )
        wrap_chat_completions_create_async(async_orcarouter_client)
        await async_orcarouter_client.chat.completions.create(
            model="orcarouter/auto", messages=[]
        )
        assert any(s.name == "ORCAROUTER_API_CALL" for s in collecting_exporter.spans)

    @pytest.mark.asyncio
    async def test_records_model_name(
        self, tracer, collecting_exporter, async_orcarouter_client
    ):
        response = make_orcarouter_response(model="orcarouter/auto")
        async_orcarouter_client.chat.completions.create = AsyncMock(
            return_value=response
        )
        wrap_chat_completions_create_async(async_orcarouter_client)
        await async_orcarouter_client.chat.completions.create(
            model="orcarouter/auto", messages=[]
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ORCAROUTER_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME)
            == "orcarouter/auto"
        )
