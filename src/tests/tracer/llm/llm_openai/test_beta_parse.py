"""Tests for OpenAI beta.chat.completions.parse wrapper."""

import pytest

pytest.importorskip("openai")

from judgeval.tracer.llm.llm_openai.wrapper import (
    wrap_openai_client_sync,
)
from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception

# All fixtures are imported automatically from conftest.py


class BaseOpenAIBetaParseTest:
    """Base class with helper methods for OpenAI beta.chat.completions.parse tests."""

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


class TestSyncBetaParse(BaseOpenAIBetaParseTest):
    def test_beta_chat_completions_parse(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test sync beta.chat.completions.parse with structured outputs and tracing verification"""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        response = sync_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "test"}],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_multiple_calls_same_client(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test multiple calls to ensure context isolation with tracing verification"""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = sync_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: first"}],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        response2 = sync_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: second"}],
            response_format=TestResponse,
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
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        with pytest.raises(Exception):
            sync_client_maybe_wrapped.beta.chat.completions.parse(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "word: test"}],
                response_format=TestResponse,
                max_completion_tokens=1000,
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)


class TestAsyncBetaParse(BaseOpenAIBetaParseTest):
    @pytest.mark.asyncio
    async def test_beta_chat_completions_parse(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async beta.chat.completions.parse with structured outputs and tracing verification"""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        response = await async_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: test"}],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test multiple async calls to ensure context isolation with tracing verification"""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = await async_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: first"}],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        response2 = await async_client_maybe_wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: second"}],
            response_format=TestResponse,
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
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        with pytest.raises(Exception):
            await async_client_maybe_wrapped.beta.chat.completions.parse(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "word: test"}],
                response_format=TestResponse,
                max_completion_tokens=1000,
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestEdgeCases(BaseOpenAIBetaParseTest):
    def test_concurrent_calls_different_clients(
        self, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test multiple wrapped clients don't interfere with tracing verification"""
        from openai import OpenAI
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        client1 = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))
        client2 = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))

        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = client1.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: one"}],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        response2 = client2.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: two"}],
            response_format=TestResponse,
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


class TestSafetyGuarantees(BaseOpenAIBetaParseTest):
    def test_safe_serialize_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test that if safe_serialize throws, user code still works with tracing verification"""
        from judgeval.utils import serialize  # type: ignore
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        def broken_serialize(obj):
            raise RuntimeError("Serialization failed!")

        monkeypatch.setattr(serialize, "safe_serialize", broken_serialize)

        wrapped_client = wrap_openai_client_sync(tracer, sync_client)
        response = wrapped_client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: test"}],
            response_format=TestResponse,
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
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        unwrapped = OpenAI(api_key=openai_api_key)
        wrapped = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))

        unwrapped_response = unwrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: test"}],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        wrapped_response = wrapped.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: test"}],
            response_format=TestResponse,
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

    def test_set_span_attribute_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test that span attribute errors don't break user code with tracing verification"""
        from judgeval.tracer import utils  # type: ignore
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        original_set = utils.set_span_attribute

        def broken_set_attribute(span, key, value):
            if "COMPLETION" in key:
                raise RuntimeError("Attribute setting failed!")
            return original_set(span, key, value)

        monkeypatch.setattr(utils, "set_span_attribute", broken_set_attribute)

        wrapped_client = wrap_openai_client_sync(tracer, sync_client)
        response = wrapped_client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "word: test"}],
            response_format=TestResponse,
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
