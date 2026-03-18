from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.instrumentation.llm.llm_google.generate_content import (
    wrap_generate_content_sync,
)
from tests.v1.instrumentation.llm.google.conftest import make_google_response


class TestGoogleGenerateContent:
    def test_creates_span(self, tracer, collecting_exporter, google_client):
        response = make_google_response()
        google_client.models.generate_content = MagicMock(return_value=response)
        wrap_generate_content_sync(google_client)
        google_client.models.generate_content(
            model="gemini-2.0-flash", contents="hello"
        )
        assert any(s.name == "GOOGLE_API_CALL" for s in collecting_exporter.spans)

    def test_span_has_llm_kind(self, tracer, collecting_exporter, google_client):
        response = make_google_response()
        google_client.models.generate_content = MagicMock(return_value=response)
        wrap_generate_content_sync(google_client)
        google_client.models.generate_content(
            model="gemini-2.0-flash", contents="hello"
        )
        span = next(s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL")
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    def test_records_token_usage(self, tracer, collecting_exporter, google_client):
        response = make_google_response(prompt_tokens=20, completion_tokens=10)
        google_client.models.generate_content = MagicMock(return_value=response)
        wrap_generate_content_sync(google_client)
        google_client.models.generate_content(
            model="gemini-2.0-flash", contents="hello"
        )
        span = next(s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL")
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 20
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 10

    def test_error_sets_error_status(self, tracer, collecting_exporter, google_client):
        google_client.models.generate_content = MagicMock(
            side_effect=RuntimeError("fail")
        )
        wrap_generate_content_sync(google_client)
        with pytest.raises(RuntimeError):
            google_client.models.generate_content(
                model="gemini-2.0-flash", contents="hello"
            )
        span = next(s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL")
        assert span.status.status_code.name == "ERROR"

    def test_returns_result(self, tracer, google_client):
        response = make_google_response()
        google_client.models.generate_content = MagicMock(return_value=response)
        wrap_generate_content_sync(google_client)
        result = google_client.models.generate_content(
            model="gemini-2.0-flash", contents="hello"
        )
        assert result is response

    def test_wrap_replaces_method(self, tracer, google_client):
        original = google_client.models.generate_content
        wrap_generate_content_sync(google_client)
        assert google_client.models.generate_content is not original
