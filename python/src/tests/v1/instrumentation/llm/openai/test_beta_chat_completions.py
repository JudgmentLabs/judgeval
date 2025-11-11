"""Tests for OpenAI beta.chat.completions.parse wrapper."""

import pytest
from pydantic import BaseModel

pytest.importorskip("openai")

from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception


class Step(BaseModel):
    explanation: str
    output: str


class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str


class BaseOpenAIBetaChatCompletionsTest:
    """Base class with helper methods for OpenAI beta.chat.completions tests."""

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


class TestSyncBetaChatCompletionsParse(BaseOpenAIBetaChatCompletionsTest):
    def test_beta_parse(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync beta.chat.completions.parse with tracing verification"""
        response = sync_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            temperature=1,
            messages=[
                {"role": "user", "content": "Solve 8x + 31 = 2. Return just the answer"}
            ],
            response_format=MathResponse,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.parsed is not None
        assert isinstance(response.choices[0].message.parsed, MathResponse)
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

        response1 = sync_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            temperature=1,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_format=MathResponse,
        )

        response2 = sync_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            temperature=1,
            messages=[{"role": "user", "content": "What is 3+3?"}],
            response_format=MathResponse,
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
            sync_client_maybe_wrapped.beta.chat.completions.parse(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "test"}],
                response_format=MathResponse,
            )

        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)


class TestAsyncBetaChatCompletionsParse(BaseOpenAIBetaChatCompletionsTest):
    @pytest.mark.asyncio
    async def test_beta_parse(self, async_client_maybe_wrapped, mock_processor):
        """Test async beta.chat.completions.parse with tracing verification"""
        response = await async_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            temperature=1,
            messages=[
                {"role": "user", "content": "Solve 8x + 31 = 2. Return just the answer"}
            ],
            response_format=MathResponse,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.parsed is not None
        assert isinstance(response.choices[0].message.parsed, MathResponse)
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

        response1 = await async_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            temperature=1,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_format=MathResponse,
        )

        response2 = await async_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            temperature=1,
            messages=[{"role": "user", "content": "What is 3+3?"}],
            response_format=MathResponse,
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
            await async_client_maybe_wrapped.beta.chat.completions.parse(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "test"}],
                response_format=MathResponse,
            )

        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)
