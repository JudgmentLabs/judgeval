"""Tests for OpenAI chat.completions wrapper."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Deprecated: Use v1 tests instead (src/tests/v1/instrumentation/llm/openai/)"
)

pytest.importorskip("openai")

from judgeval.tracer.llm.llm_openai.wrapper import (
    wrap_openai_client_sync,
)
from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception

# All fixtures are imported automatically from conftest.py


class BaseOpenAIChatCompletionsTest:
    """Base class with helper methods for OpenAI chat.completions tests."""

    def verify_tracing_if_wrapped(
        self, client, mock_processor, expected_model_name="gpt-5-nano"
    ):
        """Helper method to verify tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name=expected_model_name,
            )

    def verify_exception_if_wrapped(self, client, mock_processor):
        """Helper method to verify exception tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            assert_span_has_exception(span, "OPENAI_API_CALL")


class TestSyncChatCompletions(BaseOpenAIChatCompletionsTest):
    def test_chat_completions_create(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync chat.completions.create with tracing verification"""
        response = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_completion_tokens=1000,
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

    def test_multiple_calls_same_client(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test multiple calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_completion_tokens=1000,
        )

        response2 = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_completion_tokens=1000,
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
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )

    def test_invalid_model_name_error(self, sync_client_maybe_wrapped, mock_processor):
        """Test that invalid model names raise exceptions with tracing verification"""
        with pytest.raises(Exception):
            sync_client_maybe_wrapped.chat.completions.create(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=1000,
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)


class TestAsyncChatCompletions(BaseOpenAIChatCompletionsTest):
    @pytest.mark.asyncio
    async def test_chat_completions_create(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async chat.completions.create with tracing verification"""
        response = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_completion_tokens=1000,
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
    async def test_multiple_calls_same_client(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test multiple async calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_completion_tokens=1000,
        )

        response2 = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_completion_tokens=1000,
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
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
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
                max_completion_tokens=1000,
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_chat_completions_create_with_cache(
        self, wrapped_async_client, mock_processor
    ):
        """Test async chat.completions.create with cache and tracing verification"""
        prompt = "Explain machine learning in detail, including supervised learning, unsupervised learning, and deep learning. Provide examples of each type and explain how they work in practice."
        prompt = prompt * 100
        response = await wrapped_async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000,
        )

        response2 = await wrapped_async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000,
        )

        assert response is not None
        assert response2 is not None

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="OPENAI_API_CALL",
            expected_model_name="gpt-4o-mini",
            check_cache_read_value=True,
        )


class TestStreamingChatCompletions(BaseOpenAIChatCompletionsTest):
    def test_chat_completions_streaming(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test sync chat.completions.create with stream=True and tracing verification"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
            max_completion_tokens=1000,
        )

        chunks = list(stream)
        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "choices")

        has_usage = any(chunk.usage is not None for chunk in chunks)
        if has_usage:
            final_chunk = next((c for c in reversed(chunks) if c.usage), None)
            if final_chunk and final_chunk.usage:
                assert final_chunk.usage.prompt_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_streaming_content_accumulation(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Verify content is accumulated correctly across chunks with tracing verification"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say: Hello World"}],
            stream=True,
            max_completion_tokens=1000,
        )

        accumulated = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    accumulated += delta.content

        assert len(accumulated) > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_streaming_early_break(self, sync_client_maybe_wrapped, mock_processor):
        """Test breaking out of stream early with tracing verification"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Count to 10"}],
            stream=True,
            max_completion_tokens=1000,
        )

        first_chunk = next(iter(stream))
        assert first_chunk is not None

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_chat_completions_streaming_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async chat.completions.create with stream=True and tracing verification"""
        stream = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
            max_completion_tokens=1000,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "choices")

        has_usage = any(chunk.usage is not None for chunk in chunks)
        if has_usage:
            final_chunk = next((c for c in reversed(chunks) if c.usage), None)
            if final_chunk and final_chunk.usage:
                assert final_chunk.usage.prompt_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_streaming_content_accumulation_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Verify content is accumulated correctly across chunks with tracing verification"""
        stream = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say: Hello World"}],
            stream=True,
            max_completion_tokens=1000,
        )

        accumulated = ""
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    accumulated += delta.content

        assert len(accumulated) > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_streaming_early_break_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test breaking out of stream early with tracing verification"""
        stream = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Count to 10"}],
            stream=True,
            max_completion_tokens=1000,
        )

        first_chunk = await stream.__anext__()
        assert first_chunk is not None

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestEdgeCases(BaseOpenAIChatCompletionsTest):
    def test_concurrent_calls_different_clients(
        self, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test multiple wrapped clients don't interfere with tracing verification"""
        from openai import OpenAI

        client1 = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))
        client2 = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))

        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = client1.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say: one"}],
            max_completion_tokens=1000,
        )

        response2 = client2.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say: two"}],
            max_completion_tokens=1000,
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
            expected_span_name="OPENAI_API_CALL",
            expected_model_name="gpt-5-nano",
        )

        verify_span_attributes_comprehensive(
            span=span2,
            attrs=attrs2,
            expected_span_name="OPENAI_API_CALL",
            expected_model_name="gpt-5-nano",
        )

    def test_streaming_with_minimal_response(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test streaming with very short response and tracing verification"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say: hi"}],
            stream=True,
            max_completion_tokens=10,
        )

        chunks = list(stream)
        assert len(chunks) >= 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_async_streaming_with_minimal_response(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async streaming with very short response and tracing verification"""
        stream = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say: hi"}],
            stream=True,
            max_completion_tokens=10,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) >= 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestSafetyGuarantees(BaseOpenAIChatCompletionsTest):
    def test_safe_serialize_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test that if safe_serialize throws, user code still works with tracing verification"""
        from judgeval.utils import serialize  # type: ignore

        def broken_serialize(obj):
            raise RuntimeError("Serialization failed!")

        monkeypatch.setattr(serialize, "safe_serialize", broken_serialize)

        wrapped_client = wrap_openai_client_sync(tracer, sync_client)
        response = wrapped_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "say test"}],
            max_completion_tokens=1000,
        )

        assert response is not None
        assert response.choices
        assert response.choices[0].message.content

        # Verify tracing still works even with broken serialization
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="OPENAI_API_CALL",
            expected_model_name="gpt-5-nano",
            check_prompt=False,  # May fail due to broken serialization
            check_completion=False,  # May fail due to broken serialization
        )

    def test_wrapped_vs_unwrapped_structure(
        self, tracer, openai_api_key, mock_processor
    ):
        """Verify wrapped client behavior matches unwrapped structure with tracing verification"""
        from openai import OpenAI

        unwrapped = OpenAI(api_key=openai_api_key)
        wrapped = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))

        unwrapped_response = unwrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say exactly: test"}],
            max_completion_tokens=1000,
        )

        wrapped_response = wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say exactly: test"}],
            max_completion_tokens=1000,
        )

        assert type(unwrapped_response) is type(wrapped_response)
        assert hasattr(wrapped_response, "choices")
        assert hasattr(wrapped_response, "usage")
        assert hasattr(wrapped_response, "model")
        assert wrapped_response.model == unwrapped_response.model

        # Verify tracing works for wrapped client
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="OPENAI_API_CALL",
            expected_model_name="gpt-5-nano",
        )

    def test_streaming_exceptions_propagate(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Verify streaming exceptions propagate correctly with tracing verification"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            max_completion_tokens=1000,
        )

        first_chunk = next(iter(stream))
        assert first_chunk is not None

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_set_span_attribute_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test that span attribute errors don't break user code with tracing verification"""
        from judgeval.tracer import utils  # type: ignore

        original_set = utils.set_span_attribute

        def broken_set_attribute(span, key, value):
            if "COMPLETION" in key:
                raise RuntimeError("Attribute setting failed!")
            return original_set(span, key, value)

        monkeypatch.setattr(utils, "set_span_attribute", broken_set_attribute)

        wrapped_client = wrap_openai_client_sync(tracer, sync_client)
        response = wrapped_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "test"}],
            max_completion_tokens=1000,
        )

        assert response is not None
        assert response.choices[0].message.content

        # Verify tracing still works even with broken attribute setting
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="OPENAI_API_CALL",
            expected_model_name="gpt-5-nano",
            check_completion=False,  # May fail due to broken attribute setting
        )


class TestWithStreamingResponse(BaseOpenAIChatCompletionsTest):
    """Tests for with_streaming_response API which should bypass our wrapper."""

    def test_with_streaming_response_sync(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test that with_streaming_response.create() bypasses our wrapper and works correctly"""
        initial_span_count = len(mock_processor.ended_spans)

        with sync_client_maybe_wrapped.chat.completions.with_streaming_response.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test'"}],
            stream=True,
            max_completion_tokens=1000,
        ) as response:
            # Verify response has iter_lines method (ResponseContextManager behavior)
            assert hasattr(response, "iter_lines")

            # Read some lines to verify it works
            lines = list(response.iter_lines())
            assert len(lines) > 0

        # Verify no new spans were created (wrapper was bypassed)
        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            # Should not create a span since we bypass the wrapper
            assert len(mock_processor.ended_spans) == initial_span_count

    @pytest.mark.asyncio
    async def test_with_streaming_response_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test that async with_streaming_response.create() bypasses our wrapper and works correctly"""
        initial_span_count = len(mock_processor.ended_spans)

        async with (
            async_client_maybe_wrapped.chat.completions.with_streaming_response.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Say 'test'"}],
                stream=True,
                max_completion_tokens=1000,
            ) as response
        ):
            # Verify response has iter_lines method (ResponseContextManager behavior)
            assert hasattr(response, "iter_lines")

            # Read some lines to verify it works
            lines = []
            async for line in response.iter_lines():
                lines.append(line)
            assert len(lines) > 0

        # Verify no new spans were created (wrapper was bypassed)
        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            # Should not create a span since we bypass the wrapper
            assert len(mock_processor.ended_spans) == initial_span_count
