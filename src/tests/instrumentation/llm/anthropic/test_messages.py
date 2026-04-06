from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.instrumentation.llm.llm_anthropic.messages import (
    wrap_messages_create_sync,
    wrap_messages_create_async,
)
from tests.instrumentation.llm.anthropic.conftest import make_message


class TestSyncNonStreaming:
    def test_creates_span(self, tracer, collecting_exporter, sync_anthropic_client):
        msg = make_message()
        sync_anthropic_client.messages.create = MagicMock(return_value=msg)
        wrap_messages_create_sync(sync_anthropic_client)
        sync_anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
        )
        assert any(s.name == "ANTHROPIC_API_CALL" for s in collecting_exporter.spans)

    def test_span_has_llm_kind(
        self, tracer, collecting_exporter, sync_anthropic_client
    ):
        msg = make_message()
        sync_anthropic_client.messages.create = MagicMock(return_value=msg)
        wrap_messages_create_sync(sync_anthropic_client)
        sync_anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ANTHROPIC_API_CALL"
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    def test_records_token_usage(
        self, tracer, collecting_exporter, sync_anthropic_client
    ):
        msg = make_message(input_tokens=12, output_tokens=7)
        sync_anthropic_client.messages.create = MagicMock(return_value=msg)
        wrap_messages_create_sync(sync_anthropic_client)
        sync_anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
        )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ANTHROPIC_API_CALL"
        )
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 12
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 7

    def test_error_sets_error_status(
        self, tracer, collecting_exporter, sync_anthropic_client
    ):
        sync_anthropic_client.messages.create = MagicMock(
            side_effect=RuntimeError("api error")
        )
        wrap_messages_create_sync(sync_anthropic_client)
        with pytest.raises(RuntimeError):
            sync_anthropic_client.messages.create(
                model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
            )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ANTHROPIC_API_CALL"
        )
        assert span.status.status_code.name == "ERROR"

    def test_returns_result(self, tracer, sync_anthropic_client):
        msg = make_message()
        sync_anthropic_client.messages.create = MagicMock(return_value=msg)
        wrap_messages_create_sync(sync_anthropic_client)
        result = sync_anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
        )
        assert result is msg


class TestAsyncNonStreaming:
    @pytest.mark.asyncio
    async def test_creates_span(
        self, tracer, collecting_exporter, async_anthropic_client
    ):
        msg = make_message()
        async_anthropic_client.messages.create = AsyncMock(return_value=msg)
        wrap_messages_create_async(async_anthropic_client)
        await async_anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
        )
        assert any(s.name == "ANTHROPIC_API_CALL" for s in collecting_exporter.spans)

    @pytest.mark.asyncio
    async def test_error_sets_error_status(
        self, tracer, collecting_exporter, async_anthropic_client
    ):
        async_anthropic_client.messages.create = AsyncMock(
            side_effect=RuntimeError("fail")
        )
        wrap_messages_create_async(async_anthropic_client)
        with pytest.raises(RuntimeError):
            await async_anthropic_client.messages.create(
                model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
            )
        span = next(
            s for s in collecting_exporter.spans if s.name == "ANTHROPIC_API_CALL"
        )
        assert span.status.status_code.name == "ERROR"
