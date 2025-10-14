"""Tests for Together wrapper."""

import pytest

pytest.importorskip("together")

from judgeval.tracer.llm.llm_together.wrapper import wrap_together_client
from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception

# All fixtures are imported automatically from conftest.py


class BaseTogetherTest:
    """Base class with helper methods for Together tests."""

    def verify_tracing_if_wrapped(
        self,
        client,
        mock_processor,
        expected_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ):
        """Helper method to verify tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="TOGETHER_API_CALL",
                expected_model_name=expected_model_name,
            )

    def verify_exception_if_wrapped(self, client, mock_processor):
        """Helper method to verify exception tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            assert_span_has_exception(span, "TOGETHER_API_CALL")


class TestSyncWrapper(BaseTogetherTest):
    def test_chat_completions_create(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync chat.completions.create with tracing verification"""
        response = sync_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_tokens=100,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_chat_completions_create_streaming(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test sync streaming chat.completions.create with tracing verification"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0

        # Check that we got content in at least one chunk
        content_chunks = [
            chunk
            for chunk in chunks
            if chunk.choices and chunk.choices[0].delta.content
        ]
        assert len(content_chunks) > 0

        # Check last chunk has usage if available
        if chunks[-1].usage:
            assert chunks[-1].usage.prompt_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_multiple_calls_same_client(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test multiple calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = sync_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_tokens=50,
        )

        response2 = sync_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_tokens=50,
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
                expected_span_name="TOGETHER_API_CALL",
                expected_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="TOGETHER_API_CALL",
                expected_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )

    def test_invalid_model_name_error(self, sync_client_maybe_wrapped, mock_processor):
        """Test that invalid model names raise exceptions with tracing verification"""
        with pytest.raises(Exception):
            sync_client_maybe_wrapped.chat.completions.create(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=50,
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)


class TestAsyncWrapper(BaseTogetherTest):
    @pytest.mark.asyncio
    async def test_chat_completions_create(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async chat.completions.create with tracing verification"""
        response = await async_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_tokens=100,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_chat_completions_create_streaming(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async streaming chat.completions.create with tracing verification"""
        stream = await async_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

        # Check that we got content in at least one chunk
        content_chunks = [
            chunk
            for chunk in chunks
            if chunk.choices and chunk.choices[0].delta.content
        ]
        assert len(content_chunks) > 0

        # Check last chunk has usage if available
        if chunks[-1].usage:
            assert chunks[-1].usage.prompt_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test multiple async calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = await async_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_tokens=50,
        )

        response2 = await async_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_tokens=50,
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
                expected_span_name="TOGETHER_API_CALL",
                expected_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="TOGETHER_API_CALL",
                expected_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )

    @pytest.mark.asyncio
    async def test_invalid_model_name_error(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test that invalid model names raise exceptions with tracing verification"""
        with pytest.raises(Exception):
            await async_client_maybe_wrapped.chat.completions.create(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=50,
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestIdempotency(BaseTogetherTest):
    """Test that wrapping is idempotent and doesn't affect unwrapped clients"""

    def test_double_wrap_sync(self, tracer, sync_client, mock_processor):
        """Test that double wrapping doesn't break the client with tracing verification"""
        wrapped_once = wrap_together_client(tracer, sync_client)
        wrapped_twice = wrap_together_client(tracer, wrapped_once)

        response = wrapped_twice.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

        assert response is not None
        assert response.choices

        # Verify tracing works with double-wrapped client
        self.verify_tracing_if_wrapped(wrapped_twice, mock_processor)

    @pytest.mark.asyncio
    async def test_double_wrap_async(self, tracer, async_client, mock_processor):
        """Test that double wrapping async client doesn't break it with tracing verification"""
        wrapped_once = wrap_together_client(tracer, async_client)
        wrapped_twice = wrap_together_client(tracer, wrapped_once)

        response = await wrapped_twice.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

        assert response is not None
        assert response.choices

        # Verify tracing works with double-wrapped client
        self.verify_tracing_if_wrapped(wrapped_twice, mock_processor)
