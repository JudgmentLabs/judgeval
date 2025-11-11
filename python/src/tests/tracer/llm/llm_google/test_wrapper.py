"""Tests for Google wrapper."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Deprecated: Use v1 tests instead (src/tests/v1/instrumentation/llm/google/)"
)

pytest.importorskip("google.genai")

from judgeval.tracer.llm.llm_google.wrapper import wrap_google_client
from ..utils import verify_span_attributes_comprehensive, assert_span_has_exception

# All fixtures are imported automatically from conftest.py


class BaseGoogleTest:
    """Base class with helper methods for Google tests."""

    def verify_tracing_if_wrapped(
        self, client, mock_processor, expected_model_name="gemini-2.5-flash"
    ):
        """Helper method to verify tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            attrs = mock_processor.get_span_attributes(span)
            verify_span_attributes_comprehensive(
                span=span,
                attrs=attrs,
                expected_span_name="GOOGLE_API_CALL",
                expected_model_name=expected_model_name,
            )

    def verify_exception_if_wrapped(self, client, mock_processor):
        """Helper method to verify exception tracing only if client is wrapped."""
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            assert_span_has_exception(span, "GOOGLE_API_CALL")


class TestWrapper(BaseGoogleTest):
    def test_generate_content(self, client_maybe_wrapped, mock_processor):
        """Test generate_content with gemini-2.5-flash and tracing verification"""
        response = client_maybe_wrapped.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'test' and nothing else",
        )

        assert response is not None
        assert response.text
        assert response.usage_metadata
        assert response.usage_metadata.prompt_token_count > 0
        assert response.usage_metadata.candidates_token_count > 0

        # Verify tracing when wrapped
        self.verify_tracing_if_wrapped(client_maybe_wrapped, mock_processor)

    def test_multiple_calls_same_client(self, client_maybe_wrapped, mock_processor):
        """Test multiple calls to ensure context isolation with tracing verification"""
        # Track initial span count
        initial_span_count = len(mock_processor.ended_spans)

        response1 = client_maybe_wrapped.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'first'",
        )

        response2 = client_maybe_wrapped.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'second'",
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.text != response2.text

        # Verify tracing when wrapped - should have exactly 2 new spans
        if hasattr(client_maybe_wrapped, "_judgment_tracer"):
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
                expected_span_name="GOOGLE_API_CALL",
                expected_model_name="gemini-2.5-flash",
            )

            verify_span_attributes_comprehensive(
                span=span2,
                attrs=attrs2,
                expected_span_name="GOOGLE_API_CALL",
                expected_model_name="gemini-2.5-flash",
            )

    def test_error_recorded_in_span(self, client_maybe_wrapped, mock_processor):
        """Test that errors are properly recorded in spans with tracing verification"""
        with pytest.raises(Exception):
            client_maybe_wrapped.models.generate_content(
                model="invalid-model-name",
                contents="Test",
            )

        # Verify tracing when wrapped - should have exception recorded
        self.verify_exception_if_wrapped(client_maybe_wrapped, mock_processor)


class TestWrapperIdempotency(BaseGoogleTest):
    def test_double_wrapping(self, tracer, client, mock_processor):
        """Test that wrapping the same client twice doesn't break anything with tracing verification"""
        client1 = wrap_google_client(tracer, client)
        client2 = wrap_google_client(tracer, client1)

        response = client2.models.generate_content(
            model="gemini-2.5-flash",
            contents="Test",
        )

        assert response is not None

        # Verify tracing works with double-wrapped client
        self.verify_tracing_if_wrapped(client2, mock_processor)
