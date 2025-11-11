"""Tests for Anthropic wrapper."""

import pytest
import random
import string

pytestmark = pytest.mark.skip(
    reason="Deprecated: Use v1 tests instead (src/tests/v1/instrumentation/llm/anthropic/)"
)

pytest.importorskip("anthropic")

from judgeval.tracer.llm.llm_anthropic.wrapper import (
    wrap_anthropic_client_sync,
)
from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception

# All fixtures are imported automatically from conftest.py


class BaseAnthropicTest:
    """Base class with helper methods for Anthropic tests."""

    def verify_tracing_if_wrapped(
        self, client, mock_processor, expected_model_name="claude-3-haiku-20240307"
    ):
        """Helper method to verify tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="ANTHROPIC_API_CALL",
                expected_model_name=expected_model_name,
            )

    def verify_exception_if_wrapped(self, client, mock_processor):
        """Helper method to verify exception tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            assert_span_has_exception(span, "ANTHROPIC_API_CALL")


class TestNonStreamingSyncWrapper(BaseAnthropicTest):
    def test_messages_create(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync messages.create with tracing verification"""
        response = sync_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
        )

        assert response is not None
        assert response.content
        assert len(response.content) > 0
        assert response.model
        assert response.usage
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_multiple_calls_same_client(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test multiple calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = sync_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'first'"}],
        )

        response2 = sync_client_maybe_wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'second'"}],
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

        # Verify tracing when wrapped - should have exactly 2 new spans
        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 2

            span1 = mock_processor.ended_spans[initial_span_count]
            span2 = mock_processor.ended_spans[initial_span_count + 1]

            # Verify spans have different contexts
            assert span1.context.span_id != span2.context.span_id

            # Verify both spans have correct attributes
            attrs1 = mock_processor.get_span_attributes(span1)
            attrs2 = mock_processor.get_span_attributes(span2)

            verify_span_attributes_comprehensive(
                span=span1,
                attrs=attrs1,
                expected_span_name="ANTHROPIC_API_CALL",
                expected_model_name="claude-3-haiku-20240307",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="ANTHROPIC_API_CALL",
                expected_model_name="claude-3-5-haiku-20241022",
            )


class TestNonStreamingAsyncWrapper(BaseAnthropicTest):
    @pytest.mark.asyncio
    async def test_messages_create(self, async_client_maybe_wrapped, mock_processor):
        """Test async messages.create with tracing verification"""
        response = await async_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
        )

        assert response is not None
        assert response.content
        assert len(response.content) > 0
        assert response.model
        assert response.usage
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test multiple async calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = await async_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'first'"}],
        )

        response2 = await async_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'second'"}],
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

        # Verify tracing when wrapped - should have exactly 2 new spans
        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 2

            span1 = mock_processor.ended_spans[initial_span_count]
            span2 = mock_processor.ended_spans[initial_span_count + 1]

            # Verify spans have different contexts
            assert span1.context.span_id != span2.context.span_id

            # Verify both spans have correct attributes
            attrs1 = mock_processor.get_span_attributes(span1)
            attrs2 = mock_processor.get_span_attributes(span2)

            verify_span_attributes_comprehensive(
                span=span1,
                attrs=attrs1,
                expected_span_name="ANTHROPIC_API_CALL",
                expected_model_name="claude-3-haiku-20240307",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="ANTHROPIC_API_CALL",
                expected_model_name="claude-3-haiku-20240307",
            )

    @pytest.mark.asyncio
    async def test_messages_create_with_cache(
        self, wrapped_async_client, mock_processor
    ):
        """Test async messages.create with cache and tracing verification"""
        pride_text = "".join(random.choices(string.ascii_letters, k=100)) * 30
        system_blocks = [
            {
                "type": "text",
                "text": (
                    "You are an AI assistant tasked with analyzing literary works. "
                    "Your goal is to provide insightful commentary on themes, "
                    "characters, and writing style.\n"
                ),
            },
            {
                "type": "text",
                "text": pride_text,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        user_msg = [
            {
                "role": "user",
                "content": "Analyze the major themes in 'Pride and Prejudice'.",
            }
        ]

        response1 = await wrapped_async_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=system_blocks,
            messages=user_msg,
        )
        print(response1)

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="ANTHROPIC_API_CALL",
            expected_model_name="claude-3-haiku-20240307",
            check_cache_creation_value=True,
        )

        response2 = await wrapped_async_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=system_blocks,
            messages=user_msg,
        )

        print(response2)

        assert response1 is not None
        assert response2 is not None

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="ANTHROPIC_API_CALL",
            expected_model_name="claude-3-haiku-20240307",
            check_cache_read_value=True,
        )


class TestStreamingSync(BaseAnthropicTest):
    def test_messages_create_streaming(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync messages.create with stream=True and tracing verification"""
        stream = sync_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "type")

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_messages_stream_context_manager(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test sync messages.stream with context manager and tracing verification"""
        with sync_client_maybe_wrapped.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
        ) as stream:
            text_chunks = list(stream.text_stream)
            assert len(text_chunks) > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_streaming_content_accumulation(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Verify content is accumulated correctly across chunks with tracing verification"""
        with sync_client_maybe_wrapped.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: Hello World"}],
        ) as stream:
            accumulated = ""
            for text_chunk in stream.text_stream:
                accumulated += text_chunk

            assert len(accumulated) > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_streaming_early_break(self, sync_client_maybe_wrapped, mock_processor):
        """Test breaking out of stream early with tracing verification"""
        stream = sync_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 10"}],
            stream=True,
        )

        first_chunk = next(iter(stream))
        assert first_chunk is not None

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_streaming_early_break_with_error(self, sync_client_maybe_wrapped):
        """Test breaking out of stream early with error"""
        with pytest.raises(Exception) as exc_info:
            sync_client_maybe_wrapped.messages.create(
                model="non-existent-model",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Count to 10"}],
                stream=True,
            )

        assert exc_info.value is not None

    def test_streaming_early_break_with_error_context_manager(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test breaking out of stream early with error and tracing verification"""
        with pytest.raises(Exception) as exc_info:
            with sync_client_maybe_wrapped.messages.stream(
                model="non-existent-model",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Count to 10"}],
            ) as stream:
                for chunk in stream:
                    pass
        assert exc_info.value is not None

        # Verify tracing when wrapped - should have exception recorded
        if len(mock_processor.ended_spans) > 0:
            span = mock_processor.get_last_ended_span()
            assert_span_has_exception(span, "ANTHROPIC_API_CALL")


class TestStreamingAsync(BaseAnthropicTest):
    @pytest.mark.asyncio
    async def test_messages_create_streaming(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async messages.create with stream=True and tracing verification"""
        stream = await async_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "type")

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_messages_stream_context_manager(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async messages.stream with context manager and tracing verification"""
        async with async_client_maybe_wrapped.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
        ) as stream:
            text_chunks = [chunk async for chunk in stream.text_stream]
            assert len(text_chunks) > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_streaming_content_accumulation(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Verify content is accumulated correctly across chunks with tracing verification"""
        async with async_client_maybe_wrapped.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: Hello World"}],
        ) as stream:
            accumulated = ""
            async for text_chunk in stream.text_stream:
                accumulated += text_chunk

            assert len(accumulated) > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_streaming_early_break(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test breaking out of stream early with tracing verification"""
        stream = await async_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 10"}],
            stream=True,
        )

        first_chunk = await stream.__anext__()
        assert first_chunk is not None

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_streaming_early_break_with_error(self, async_client_maybe_wrapped):
        """Test breaking out of stream early with error"""
        with pytest.raises(Exception) as exc_info:
            await async_client_maybe_wrapped.messages.create(
                model="non-existent-model",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Count to 10"}],
                stream=True,
            )
        assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_streaming_early_break_with_error_context_manager(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test breaking out of stream early with error and tracing verification"""
        with pytest.raises(Exception) as exc_info:
            with async_client_maybe_wrapped.messages.stream(
                model="non-existent-model",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Count to 10"}],
            ) as stream:
                async for chunk in stream:
                    pass
        assert exc_info.value is not None

        # Verify tracing when wrapped - should have exception recorded
        if len(mock_processor.ended_spans) > 0:
            span = mock_processor.get_last_ended_span()
            assert_span_has_exception(span, "ANTHROPIC_API_CALL")


class TestEdgeCases(BaseAnthropicTest):
    def test_concurrent_calls_different_clients(
        self, tracer, sync_client, anthropic_api_key, mock_processor
    ):
        """Test multiple wrapped clients don't interfere with tracing verification"""
        from anthropic import Anthropic

        client1 = wrap_anthropic_client_sync(
            tracer, Anthropic(api_key=anthropic_api_key)
        )
        client2 = wrap_anthropic_client_sync(
            tracer, Anthropic(api_key=anthropic_api_key)
        )

        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = client1.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: one"}],
        )

        response2 = client2.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: two"}],
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

        # Verify tracing - should have exactly 2 new spans from different clients
        assert len(mock_processor.ended_spans) == initial_span_count + 2

        span1 = mock_processor.ended_spans[initial_span_count]
        span2 = mock_processor.ended_spans[initial_span_count + 1]

        # Verify spans have different contexts
        assert span1.context.span_id != span2.context.span_id

        # Verify both spans have correct attributes
        attrs1 = mock_processor.get_span_attributes(span1)
        attrs2 = mock_processor.get_span_attributes(span2)

        verify_span_attributes_comprehensive(
            span=span1,
            attrs=attrs1,
            expected_span_name="ANTHROPIC_API_CALL",
            expected_model_name="claude-3-haiku-20240307",
        )

        verify_span_attributes_comprehensive(
            span=span2,
            attrs=attrs2,
            expected_span_name="ANTHROPIC_API_CALL",
            expected_model_name="claude-3-haiku-20240307",
        )


class TestSafetyGuarantees(BaseAnthropicTest):
    def test_safe_serialize_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, anthropic_api_key, mock_processor
    ):
        """Test that if safe_serialize throws, user code still works with tracing verification"""
        from judgeval.utils import serialize  # type: ignore

        def broken_serialize(obj):
            raise RuntimeError("Serialization failed!")

        monkeypatch.setattr(serialize, "safe_serialize", broken_serialize)

        wrapped_client = wrap_anthropic_client_sync(tracer, sync_client)
        response = wrapped_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
        )

        assert response is not None
        assert response.content
        assert len(response.content) > 0

        # Verify tracing still works even with broken serialization
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="ANTHROPIC_API_CALL",
            expected_model_name="claude-3-haiku-20240307",
            check_prompt=False,  # May fail due to broken serialization
            check_completion=False,  # May fail due to broken serialization
        )

    def test_wrapped_vs_unwrapped_structure(
        self, tracer, anthropic_api_key, mock_processor
    ):
        """Verify wrapped client behavior matches unwrapped structure with tracing verification"""
        from anthropic import Anthropic

        unwrapped = Anthropic(api_key=anthropic_api_key)
        wrapped = wrap_anthropic_client_sync(
            tracer, Anthropic(api_key=anthropic_api_key)
        )

        unwrapped_response = unwrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say exactly: test"}],
        )

        wrapped_response = wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say exactly: test"}],
        )

        assert type(unwrapped_response) is type(wrapped_response)
        assert hasattr(wrapped_response, "content")
        assert hasattr(wrapped_response, "usage")
        assert hasattr(wrapped_response, "model")
        assert wrapped_response.model == unwrapped_response.model

    def test_exceptions_propagate_correctly(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Verify API exceptions still reach user with tracing verification"""
        with pytest.raises(Exception) as exc_info:
            sync_client_maybe_wrapped.messages.create(
                model="invalid-model-name-that-does-not-exist",
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value is not None

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_async_exceptions_propagate(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Verify async API exceptions still reach user with tracing verification"""
        with pytest.raises(Exception) as exc_info:
            await async_client_maybe_wrapped.messages.create(
                model="invalid-model-name-that-does-not-exist",
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value is not None

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)

    def test_streaming_exceptions_propagate(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Verify streaming exceptions propagate correctly with tracing verification"""
        stream = sync_client_maybe_wrapped.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
            stream=True,
        )

        first_chunk = next(iter(stream))
        assert first_chunk is not None

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_set_span_attribute_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, anthropic_api_key, mock_processor
    ):
        """Test that span attribute errors don't break user code with tracing verification"""
        from judgeval.tracer import utils  # type: ignore

        original_set = utils.set_span_attribute

        def broken_set_attribute(span, key, value):
            if "COMPLETION" in key:
                raise RuntimeError("Attribute setting failed!")
            return original_set(span, key, value)

        monkeypatch.setattr(utils, "set_span_attribute", broken_set_attribute)

        wrapped_client = wrap_anthropic_client_sync(tracer, sync_client)
        response = wrapped_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
        )

        assert response is not None
        assert response.content

        # Verify tracing still works even with broken attribute setting
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="ANTHROPIC_API_CALL",
            expected_model_name="claude-3-haiku-20240307",
            check_completion=False,  # May fail due to broken attribute setting
        )
