from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.instrumentation.llm.llm_google.generate_content import (
    wrap_generate_content_stream_sync,
)


def make_google_stream_chunk(text=None, prompt_tokens=None, completion_tokens=None):
    chunk = MagicMock()
    chunk.text = text
    chunk.model_version = None
    if prompt_tokens is not None or completion_tokens is not None:
        chunk.usage_metadata = MagicMock(
            prompt_token_count=prompt_tokens or 0,
            candidates_token_count=completion_tokens or 0,
            cached_content_token_count=None,
        )
    else:
        chunk.usage_metadata = None
    return chunk


class TestGoogleGenerateContentStream:
    def test_creates_span(self, tracer, collecting_exporter, google_client):
        chunks = [make_google_stream_chunk(text="Hi")]
        google_client.models.generate_content_stream = MagicMock(
            return_value=iter(chunks)
        )
        wrap_generate_content_stream_sync(google_client)
        list(google_client.models.generate_content_stream(model="gemini-2.0-flash", contents="hello"))
        assert any(s.name == "GOOGLE_API_CALL" for s in collecting_exporter.spans)

    def test_span_has_llm_kind(self, tracer, collecting_exporter, google_client):
        chunks = [make_google_stream_chunk(text="Hi")]
        google_client.models.generate_content_stream = MagicMock(
            return_value=iter(chunks)
        )
        wrap_generate_content_stream_sync(google_client)
        list(google_client.models.generate_content_stream(model="gemini-2.0-flash", contents="hello"))
        span = next(s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL")
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    def test_accumulates_text_across_chunks(
        self, tracer, collecting_exporter, google_client
    ):
        chunks = [
            make_google_stream_chunk(text="Hello "),
            make_google_stream_chunk(text="world"),
            make_google_stream_chunk(text=None, prompt_tokens=8, completion_tokens=2),
        ]
        google_client.models.generate_content_stream = MagicMock(
            return_value=iter(chunks)
        )
        wrap_generate_content_stream_sync(google_client)
        list(google_client.models.generate_content_stream(model="gemini-2.0-flash", contents="hello"))
        span = next(s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL")
        assert span.attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Hello world"

    def test_records_token_usage_from_final_chunk(
        self, tracer, collecting_exporter, google_client
    ):
        chunks = [
            make_google_stream_chunk(text="A"),
            make_google_stream_chunk(text="B", prompt_tokens=15, completion_tokens=7),
        ]
        google_client.models.generate_content_stream = MagicMock(
            return_value=iter(chunks)
        )
        wrap_generate_content_stream_sync(google_client)
        list(google_client.models.generate_content_stream(model="gemini-2.0-flash", contents="hello"))
        span = next(s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL")
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 15
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 7

    def test_chunks_without_text_do_not_add_to_completion(
        self, tracer, collecting_exporter, google_client
    ):
        chunks = [
            make_google_stream_chunk(text=None),
            make_google_stream_chunk(text="Only this"),
        ]
        google_client.models.generate_content_stream = MagicMock(
            return_value=iter(chunks)
        )
        wrap_generate_content_stream_sync(google_client)
        list(google_client.models.generate_content_stream(model="gemini-2.0-flash", contents="hello"))
        span = next(s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL")
        assert span.attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Only this"

    def test_error_sets_error_status(
        self, tracer, collecting_exporter, google_client
    ):
        def failing_stream(*args, **kwargs):
            yield make_google_stream_chunk(text="A")
            raise RuntimeError("stream failed")

        google_client.models.generate_content_stream = failing_stream
        wrap_generate_content_stream_sync(google_client)
        with pytest.raises(RuntimeError):
            list(google_client.models.generate_content_stream(model="gemini-2.0-flash", contents="hello"))
        span = next(s for s in collecting_exporter.spans if s.name == "GOOGLE_API_CALL")
        assert span.status.status_code.name == "ERROR"

    def test_wrap_replaces_method(self, tracer, google_client):
        original = google_client.models.generate_content_stream
        wrap_generate_content_stream_sync(google_client)
        assert google_client.models.generate_content_stream is not original

    def test_yields_all_chunks_unchanged(self, tracer, google_client):
        chunk1 = make_google_stream_chunk(text="A")
        chunk2 = make_google_stream_chunk(text="B")
        google_client.models.generate_content_stream = MagicMock(
            return_value=iter([chunk1, chunk2])
        )
        wrap_generate_content_stream_sync(google_client)
        result = list(
            google_client.models.generate_content_stream(
                model="gemini-2.0-flash", contents="hello"
            )
        )
        assert result == [chunk1, chunk2]
