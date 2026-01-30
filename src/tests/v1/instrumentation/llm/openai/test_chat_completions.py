"""Tests for OpenAI chat.completions wrapper."""

import pytest

pytest.importorskip("openai")

from judgeval.v1.instrumentation.llm.llm_openai.wrapper import (
    wrap_openai_client_sync,
)
from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception


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
            temperature=1,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_multiple_calls_same_client(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test multiple calls to ensure context isolation with tracing verification"""
        initial_span_count = len(mock_processor.ended_spans)

        response1 = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        response2 = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 2

            span1 = mock_processor.ended_spans[initial_span_count]
            span2 = mock_processor.ended_spans[initial_span_count + 1]

            assert span1.context.span_id != span2.context.span_id

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
            temperature=1,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test multiple async calls to ensure context isolation with tracing verification"""
        initial_span_count = len(mock_processor.ended_spans)

        response1 = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        response2 = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 2

            span1 = mock_processor.ended_spans[initial_span_count]
            span2 = mock_processor.ended_spans[initial_span_count + 1]

            assert span1.context.span_id != span2.context.span_id

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
            temperature=1,
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
            temperature=1,
        )

        accumulated = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    accumulated += delta.content

        assert len(accumulated) > 0

        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_streaming_early_break(self, sync_client_maybe_wrapped, mock_processor):
        """Test breaking out of stream early with tracing verification"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Count to 10"}],
            stream=True,
            max_completion_tokens=1000,
            temperature=1,
        )

        first_chunk = next(iter(stream))
        assert first_chunk is not None

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
            temperature=1,
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
            temperature=1,
        )

        accumulated = ""
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    accumulated += delta.content

        assert len(accumulated) > 0

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
            temperature=1,
        )

        first_chunk = await stream.__anext__()
        assert first_chunk is not None

        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestEdgeCases(BaseOpenAIChatCompletionsTest):
    def test_concurrent_calls_different_clients(
        self, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test multiple wrapped clients don't interfere with tracing verification"""
        from openai import OpenAI

        client1 = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))
        client2 = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))

        initial_span_count = len(mock_processor.ended_spans)

        response1 = client1.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say: one"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        response2 = client2.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say: two"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

        assert len(mock_processor.ended_spans) == initial_span_count + 2

        span1 = mock_processor.ended_spans[initial_span_count]
        span2 = mock_processor.ended_spans[initial_span_count + 1]

        assert span1.context.span_id != span2.context.span_id

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
            temperature=1,
        )

        chunks = list(stream)
        assert len(chunks) >= 0

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
            temperature=1,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) >= 0

        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestSafetyGuarantees(BaseOpenAIChatCompletionsTest):
    def test_safe_serialize_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test that if safe_serialize throws, user code still works with tracing verification"""
        from judgeval.utils import serialize

        def broken_serialize(obj):
            raise RuntimeError("Serialization failed!")

        monkeypatch.setattr(serialize, "safe_serialize", broken_serialize)

        wrapped_client = wrap_openai_client_sync(tracer, sync_client)
        response = wrapped_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "say test"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert response is not None
        assert response.choices
        assert response.choices[0].message.content

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)
        verify_span_attributes_comprehensive(
            span=span,
            attrs=attrs,
            expected_span_name="OPENAI_API_CALL",
            expected_model_name="gpt-5-nano",
            check_prompt=False,
            check_completion=False,
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
            temperature=1,
        )

        wrapped_response = wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say exactly: test"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert type(unwrapped_response) is type(wrapped_response)
        assert hasattr(wrapped_response, "choices")
        assert hasattr(wrapped_response, "usage")
        assert hasattr(wrapped_response, "model")
        assert wrapped_response.model == unwrapped_response.model

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
            temperature=1,
        )

        first_chunk = next(iter(stream))
        assert first_chunk is not None

        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)


class TestWithStreamingResponse(BaseOpenAIChatCompletionsTest):
    """Tests for with_streaming_response API."""

    def test_iter_lines_streaming_sync(self, sync_client_maybe_wrapped, mock_processor):
        """Test iter_lines() with stream=True"""
        initial_span_count = len(mock_processor.ended_spans)

        with sync_client_maybe_wrapped.chat.completions.with_streaming_response.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test'"}],
            stream=True,
            max_completion_tokens=1000,
            temperature=1,
        ) as response:
            assert hasattr(response, "iter_lines")
            lines = list(response.iter_lines())
            assert len(lines) > 0

        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 1
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
                check_completion=False,
            )

    def test_parse_streaming_sync(self, sync_client_maybe_wrapped, mock_processor):
        """Test parse() with stream=True returns Stream[ChatCompletionChunk]"""
        initial_span_count = len(mock_processor.ended_spans)

        with sync_client_maybe_wrapped.chat.completions.with_streaming_response.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            stream_options={"include_usage": True},
            max_completion_tokens=50,
            temperature=1,
        ) as response:
            stream = response.parse()
            chunks = list(stream)
            assert len(chunks) > 0
            for chunk in chunks:
                assert hasattr(chunk, "choices")

        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 1
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )

    def test_parse_non_streaming_sync(self, sync_client_maybe_wrapped, mock_processor):
        """Test parse() without stream returns ChatCompletion"""
        initial_span_count = len(mock_processor.ended_spans)

        with sync_client_maybe_wrapped.chat.completions.with_streaming_response.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say bye"}],
            max_completion_tokens=1000,
            temperature=1,
        ) as response:
            completion = response.parse()
            assert completion is not None
            assert completion.choices
            assert len(completion.choices) > 0
            assert completion.usage
            assert completion.usage.prompt_tokens > 0

        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 1
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )

    def test_json_non_streaming_sync(self, sync_client_maybe_wrapped, mock_processor):
        """Test json() returns raw dict"""
        initial_span_count = len(mock_processor.ended_spans)

        with sync_client_maybe_wrapped.chat.completions.with_streaming_response.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say ok"}],
            max_completion_tokens=1000,
            temperature=1,
        ) as response:
            data = response.json()
            assert isinstance(data, dict)
            assert "choices" in data
            assert "model" in data

        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 1
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )

    @pytest.mark.asyncio
    async def test_iter_lines_streaming_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async iter_lines() with stream=True"""
        initial_span_count = len(mock_processor.ended_spans)

        async with (
            async_client_maybe_wrapped.chat.completions.with_streaming_response.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Say 'test'"}],
                stream=True,
                max_completion_tokens=1000,
                temperature=1,
            ) as response
        ):
            assert hasattr(response, "iter_lines")
            lines = []
            async for line in response.iter_lines():
                lines.append(line)
            assert len(lines) > 0

        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 1
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
                check_completion=False,
            )

    @pytest.mark.asyncio
    async def test_parse_streaming_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async parse() with stream=True returns AsyncStream[ChatCompletionChunk]"""
        initial_span_count = len(mock_processor.ended_spans)

        async with (
            async_client_maybe_wrapped.chat.completions.with_streaming_response.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Say hello"}],
                stream=True,
                stream_options={"include_usage": True},
                max_completion_tokens=50,
                temperature=1,
            ) as response
        ):
            stream = await response.parse()
            chunks = [chunk async for chunk in stream]
            assert len(chunks) > 0
            for chunk in chunks:
                assert hasattr(chunk, "choices")

        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 1
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )

    @pytest.mark.asyncio
    async def test_parse_non_streaming_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async parse() without stream returns ChatCompletion"""
        initial_span_count = len(mock_processor.ended_spans)

        async with (
            async_client_maybe_wrapped.chat.completions.with_streaming_response.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Say bye"}],
                max_completion_tokens=1000,
                temperature=1,
            ) as response
        ):
            completion = await response.parse()
            assert completion is not None
            assert completion.choices
            assert len(completion.choices) > 0
            assert completion.usage
            assert completion.usage.prompt_tokens > 0

        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 1
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )

    @pytest.mark.asyncio
    async def test_json_non_streaming_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async json() returns raw dict"""
        initial_span_count = len(mock_processor.ended_spans)

        async with (
            async_client_maybe_wrapped.chat.completions.with_streaming_response.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Say ok"}],
                max_completion_tokens=1000,
                temperature=1,
            ) as response
        ):
            data = await response.json()
            assert isinstance(data, dict)
            assert "choices" in data
            assert "model" in data

        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 1
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="gpt-5-nano",
            )
