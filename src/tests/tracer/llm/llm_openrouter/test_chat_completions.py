"""Tests for OpenRouter chat.completions wrapper (via OpenAI SDK)."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Deprecated: Use v1 tests instead (src/tests/v1/instrumentation/llm/openrouter/)"
)

pytest.importorskip("openai")

from judgeval.tracer.keys import AttributeKeys
from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception

# All fixtures are imported automatically from conftest.py


class BaseOpenRouterChatCompletionsTest:
    """Base class with helper methods for OpenRouter chat.completions tests."""

    def verify_tracing_if_wrapped(
        self, client, mock_processor, expected_model_name="openai/gpt-4o-mini"
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
            # Verify usage attributes are present and valid
            self.verify_usage_attributes(attrs)

    def verify_usage_attributes(self, attrs):
        """Verify all usage attributes including cost."""
        # Check token attributes
        assert AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS in attrs
        assert AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS in attrs
        assert AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS in attrs
        assert AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS in attrs
        assert AttributeKeys.JUDGMENT_USAGE_METADATA in attrs

        # Verify token values are non-negative
        assert attrs[AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS] >= 0
        assert attrs[AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS] >= 0
        assert attrs[AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS] >= 0
        assert attrs[AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS] >= 0

        # Check for cost attribute (OpenRouter provides this)
        if AttributeKeys.JUDGMENT_USAGE_TOTAL_COST_USD in attrs:
            cost = attrs[AttributeKeys.JUDGMENT_USAGE_TOTAL_COST_USD]
            assert cost >= 0, f"Cost should be non-negative, got {cost}"

    def verify_exception_if_wrapped(self, client, mock_processor):
        """Helper method to verify exception tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            assert_span_has_exception(span, "OPENAI_API_CALL")


class TestSyncChatCompletions(BaseOpenRouterChatCompletionsTest):
    def test_chat_completions_create(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync chat.completions.create via OpenRouter with tracing verification"""
        response = sync_client_maybe_wrapped.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_tokens=50,
            extra_body={"usage": {"include": True}},
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
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_tokens=50,
            extra_body={"usage": {"include": True}},
        )

        response2 = sync_client_maybe_wrapped.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_tokens=50,
            extra_body={"usage": {"include": True}},
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
                expected_model_name="openai/gpt-4o-mini",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="openai/gpt-4o-mini",
            )

    def test_invalid_model_name_error(self, sync_client_maybe_wrapped, mock_processor):
        """Test that invalid model names raise exceptions with tracing verification"""
        with pytest.raises(Exception):
            sync_client_maybe_wrapped.chat.completions.create(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=50,
                extra_body={"usage": {"include": True}},
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_different_models(self, sync_client_maybe_wrapped, mock_processor):
        """Test different OpenRouter models"""
        # Test with Anthropic model via OpenRouter
        response = sync_client_maybe_wrapped.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=50,
            extra_body={"usage": {"include": True}},
        )

        assert response is not None
        assert response.choices[0].message.content
        assert response.usage

        # Verify tracing for Anthropic model
        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="anthropic/claude-3-haiku",
            )


class TestAsyncChatCompletions(BaseOpenRouterChatCompletionsTest):
    @pytest.mark.asyncio
    async def test_chat_completions_create(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async chat.completions.create via OpenRouter with tracing verification"""
        response = await async_client_maybe_wrapped.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_tokens=50,
            extra_body={"usage": {"include": True}},
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
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_tokens=50,
            extra_body={"usage": {"include": True}},
        )

        response2 = await async_client_maybe_wrapped.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_tokens=50,
            extra_body={"usage": {"include": True}},
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
                expected_model_name="openai/gpt-4o-mini",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="OPENAI_API_CALL",
                expected_model_name="openai/gpt-4o-mini",
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
                extra_body={"usage": {"include": True}},
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestStreamingChatCompletions(BaseOpenRouterChatCompletionsTest):
    def test_streaming_chat_completions(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test sync streaming chat.completions via OpenRouter"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            stream=True,
            extra_body={"usage": {"include": True}},
        )

        chunks = list(stream)
        assert len(chunks) > 0

        # Collect content
        content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        assert len(content) > 0

        # Verify tracing when wrapped
        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            # For streaming, we verify the span exists and has correct type
            assert span.name == "OPENAI_API_CALL"
            assert attrs.get("judgment.span_kind") == "llm"

    @pytest.mark.asyncio
    async def test_streaming_chat_completions_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async streaming chat.completions via OpenRouter"""
        stream = await async_client_maybe_wrapped.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            stream=True,
            extra_body={"usage": {"include": True}},
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

        # Collect content
        content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        assert len(content) > 0

        # Verify tracing when wrapped
        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            # For streaming, we verify the span exists and has correct type
            assert span.name == "OPENAI_API_CALL"
            assert attrs.get("judgment.span_kind") == "llm"
