"""Tests for OpenAI responses wrapper."""

import pytest

pytest.importorskip("openai")

from judgeval.v1.instrumentation.llm.llm_openai.wrapper import (
    wrap_openai_client_sync,
)
from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception

# All fixtures are imported automatically from conftest.py


class BaseOpenAIResponsesTest:
    """Base class with helper methods for OpenAI responses tests."""

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


class TestSyncResponses(BaseOpenAIResponsesTest):
    def test_responses_create(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync responses.create with tracing verification"""
        response = sync_client_maybe_wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say 'test' and nothing else",
        )

        assert response is not None
        assert response.model
        assert hasattr(response, "output") or hasattr(response, "text")

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_multiple_calls_same_client(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        """Test multiple calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = sync_client_maybe_wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say 'first'",
        )

        response2 = sync_client_maybe_wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say 'second'",
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
            sync_client_maybe_wrapped.responses.create(
                model="invalid-model-name-that-does-not-exist",
                input="test",
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)


class TestAsyncResponses(BaseOpenAIResponsesTest):
    @pytest.mark.asyncio
    async def test_responses_create(self, async_client_maybe_wrapped, mock_processor):
        """Test async responses.create with tracing verification"""
        response = await async_client_maybe_wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say 'test' and nothing else",
        )

        assert response is not None
        assert response.model
        assert hasattr(response, "output") or hasattr(response, "text")

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test multiple async calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = await async_client_maybe_wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say 'first'",
        )

        response2 = await async_client_maybe_wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say 'second'",
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
            await async_client_maybe_wrapped.responses.create(
                model="invalid-model-name-that-does-not-exist",
                input="test",
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.skip(reason="Cache reading tokens does seem to be consistently set")
    @pytest.mark.asyncio
    async def test_responses_create_with_cache(
        self, wrapped_async_client, mock_processor
    ):
        """Test async responses.create with cache and tracing verification"""
        prompt = "Explain machine learning in detail, including supervised learning, unsupervised learning, and deep learning. Provide examples of each type and explain how they work in practice."
        prompt = prompt * 100
        response = await wrapped_async_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )

        response2 = await wrapped_async_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
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


class TestStreamingResponses(BaseOpenAIResponsesTest):
    def test_responses_streaming(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync responses.create with stream=True and tracing verification"""
        stream = sync_client_maybe_wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Count to 3",
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0

        for chunk in chunks:
            assert (
                hasattr(chunk, "type")
                or hasattr(chunk, "response")
                or hasattr(chunk, "id")
            )

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_responses_streaming_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async responses.create with stream=True and tracing verification"""
        stream = await async_client_maybe_wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Count to 3",
            stream=True,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) > 0

        for chunk in chunks:
            assert (
                hasattr(chunk, "type")
                or hasattr(chunk, "response")
                or hasattr(chunk, "id")
            )

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestEdgeCases(BaseOpenAIResponsesTest):
    def test_concurrent_calls_different_clients(
        self, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test multiple wrapped clients don't interfere with tracing verification"""
        from openai import OpenAI

        client1 = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))
        client2 = wrap_openai_client_sync(tracer, OpenAI(api_key=openai_api_key))

        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = client1.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say: one",
        )

        response2 = client2.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say: two",
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


class TestWithStreamingResponse(BaseOpenAIResponsesTest):
    """Tests for responses.with_streaming_response.create"""

    def test_iter_lines_streaming_sync(self, sync_client_maybe_wrapped, mock_processor):
        """Test iter_lines() with stream=True"""
        initial_span_count = len(mock_processor.ended_spans)

        with sync_client_maybe_wrapped.responses.with_streaming_response.create(
            model="gpt-5-nano",
            input="Say 'test'",
            stream=True,
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
        """Test parse() with stream=True returns streaming response"""
        initial_span_count = len(mock_processor.ended_spans)

        with sync_client_maybe_wrapped.responses.with_streaming_response.create(
            model="gpt-5-nano",
            input="Say hello",
            stream=True,
        ) as response:
            stream = response.parse()
            # Stream should be iterable
            chunks = list(stream)
            assert len(chunks) > 0

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

    def test_parse_non_streaming_sync(self, sync_client_maybe_wrapped, mock_processor):
        """Test parse() without stream returns Response object"""
        initial_span_count = len(mock_processor.ended_spans)

        with sync_client_maybe_wrapped.responses.with_streaming_response.create(
            model="gpt-5-nano",
            input="Say bye",
        ) as response:
            result = response.parse()
            assert result is not None
            assert hasattr(result, "id") or hasattr(result, "output")

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

        with sync_client_maybe_wrapped.responses.with_streaming_response.create(
            model="gpt-5-nano",
            input="Say ok",
        ) as response:
            data = response.json()
            assert isinstance(data, dict)
            assert "id" in data or "output" in data

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

        async with async_client_maybe_wrapped.responses.with_streaming_response.create(
            model="gpt-5-nano",
            input="Say 'test'",
            stream=True,
        ) as response:
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
        """Test async parse() with stream=True returns streaming response"""
        initial_span_count = len(mock_processor.ended_spans)

        async with async_client_maybe_wrapped.responses.with_streaming_response.create(
            model="gpt-5-nano",
            input="Say hello",
            stream=True,
        ) as response:
            stream = await response.parse()
            chunks = [chunk async for chunk in stream]
            assert len(chunks) > 0

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
    async def test_parse_non_streaming_async(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async parse() without stream returns Response object"""
        initial_span_count = len(mock_processor.ended_spans)

        async with async_client_maybe_wrapped.responses.with_streaming_response.create(
            model="gpt-5-nano",
            input="Say bye",
        ) as response:
            result = await response.parse()
            assert result is not None
            assert hasattr(result, "id") or hasattr(result, "output")

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

        async with async_client_maybe_wrapped.responses.with_streaming_response.create(
            model="gpt-5-nano",
            input="Say ok",
        ) as response:
            data = await response.json()
            assert isinstance(data, dict)
            assert "id" in data or "output" in data

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

    def test_sync_error_handling(self, sync_client_maybe_wrapped, mock_processor):
        """Test sync error handling for invalid model"""
        with pytest.raises(Exception):
            with sync_client_maybe_wrapped.responses.with_streaming_response.create(
                model="invalid-model-that-does-not-exist",
                input="test",
            ) as response:
                _ = response.json()

        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_async_error_handling(
        self, async_client_maybe_wrapped, mock_processor
    ):
        """Test async error handling for invalid model"""
        with pytest.raises(Exception):
            async with (
                async_client_maybe_wrapped.responses.with_streaming_response.create(
                    model="invalid-model-that-does-not-exist",
                    input="test",
                ) as response
            ):
                _ = await response.json()

        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)


class TestSafetyGuarantees(BaseOpenAIResponsesTest):
    def test_safe_serialize_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, openai_api_key, mock_processor
    ):
        """Test that if safe_serialize throws, user code still works with tracing verification"""
        from judgeval.utils import serialize  # type: ignore

        def broken_serialize(obj):
            raise RuntimeError("Serialization failed!")

        monkeypatch.setattr(serialize, "safe_serialize", broken_serialize)

        wrapped_client = wrap_openai_client_sync(tracer, sync_client)
        response = wrapped_client.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="test",
        )

        assert response is not None

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

        unwrapped_response = unwrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say exactly: test",
        )

        wrapped_response = wrapped.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="Say exactly: test",
        )

        assert type(unwrapped_response) is type(wrapped_response)
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

        original_set = utils.set_span_attribute

        def broken_set_attribute(span, key, value):
            if "COMPLETION" in key:
                raise RuntimeError("Attribute setting failed!")
            return original_set(span, key, value)

        monkeypatch.setattr(utils, "set_span_attribute", broken_set_attribute)

        wrapped_client = wrap_openai_client_sync(tracer, sync_client)
        response = wrapped_client.responses.create(
            model="gpt-5-nano",
            temperature=1,
            input="test",
        )

        assert response is not None

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
