from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.instrumentation.llm.llm_anthropic.messages_stream import (
    wrap_messages_stream_sync,
    wrap_messages_stream_async,
)
from tests.instrumentation.llm.anthropic.conftest import make_message


def _make_sync_manager(input_tokens=10, output_tokens=5):
    msg = make_message(input_tokens=input_tokens, output_tokens=output_tokens)
    stream = MagicMock()
    stream.text_stream = iter([])
    stream.get_final_message = MagicMock(return_value=msg)
    manager = MagicMock()
    manager.__enter__ = MagicMock(return_value=stream)
    manager.__exit__ = MagicMock(return_value=False)
    return manager


class TestSyncStreamWrapper:
    def test_span_is_created(self, tracer, collecting_exporter, sync_anthropic_client):
        sync_anthropic_client.messages.stream = MagicMock(return_value=_make_sync_manager())
        wrap_messages_stream_sync(sync_anthropic_client)
        with sync_anthropic_client.messages.stream(
            model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
        ):
            pass
        assert any(s.name == "ANTHROPIC_API_CALL" for s in collecting_exporter.spans)

    def test_span_records_token_usage(self, tracer, collecting_exporter, sync_anthropic_client):
        sync_anthropic_client.messages.stream = MagicMock(return_value=_make_sync_manager(input_tokens=15, output_tokens=8))
        wrap_messages_stream_sync(sync_anthropic_client)
        with sync_anthropic_client.messages.stream(
            model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
        ):
            pass
        span = next(s for s in collecting_exporter.spans if s.name == "ANTHROPIC_API_CALL")
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS) == 15
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 8

    def test_span_ends_even_when_manager_exit_raises(self, tracer, collecting_exporter, sync_anthropic_client):
        """span.end() must be called even if the underlying manager.__exit__ raises."""
        msg = make_message()
        stream = MagicMock()
        stream.text_stream = iter([])
        stream.get_final_message = MagicMock(return_value=msg)
        manager = MagicMock()
        manager.__enter__ = MagicMock(return_value=stream)
        manager.__exit__ = MagicMock(side_effect=RuntimeError("stream closed abnormally"))
        sync_anthropic_client.messages.stream = MagicMock(return_value=manager)
        wrap_messages_stream_sync(sync_anthropic_client)
        with pytest.raises(RuntimeError, match="stream closed abnormally"):
            with sync_anthropic_client.messages.stream(
                model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
            ):
                pass
        assert any(s.name == "ANTHROPIC_API_CALL" for s in collecting_exporter.spans)


class TestAsyncStreamWrapper:
    @pytest.mark.asyncio
    async def test_span_is_created(self, tracer, collecting_exporter, async_anthropic_client):
        msg = make_message()
        stream = AsyncMock()
        stream.text_stream = self._aiter([])
        stream.get_final_message = AsyncMock(return_value=msg)
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=stream)
        manager.__aexit__ = AsyncMock(return_value=False)
        async_anthropic_client.messages.stream = MagicMock(return_value=manager)
        wrap_messages_stream_async(async_anthropic_client)
        async with async_anthropic_client.messages.stream(
            model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
        ):
            pass
        assert any(s.name == "ANTHROPIC_API_CALL" for s in collecting_exporter.spans)

    @pytest.mark.asyncio
    async def test_span_ends_even_when_manager_aexit_raises(self, tracer, collecting_exporter, async_anthropic_client):
        """span.end() must be called even if the underlying manager.__aexit__ raises."""
        msg = make_message()
        stream = AsyncMock()
        stream.text_stream = self._aiter([])
        stream.get_final_message = AsyncMock(return_value=msg)
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=stream)
        manager.__aexit__ = AsyncMock(side_effect=RuntimeError("async stream closed abnormally"))
        async_anthropic_client.messages.stream = MagicMock(return_value=manager)
        wrap_messages_stream_async(async_anthropic_client)
        with pytest.raises(RuntimeError, match="async stream closed abnormally"):
            async with async_anthropic_client.messages.stream(
                model="claude-3-5-sonnet-latest", messages=[], max_tokens=100
            ):
                pass
        assert any(s.name == "ANTHROPIC_API_CALL" for s in collecting_exporter.spans)

    @staticmethod
    async def _aiter(items):
        for item in items:
            yield item
