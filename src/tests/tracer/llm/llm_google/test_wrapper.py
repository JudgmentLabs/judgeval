import pytest
import os
from typing import Any, Optional

pytest.importorskip("google.genai")

from google.genai import Client
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from judgeval.tracer.llm.llm_google.wrapper import wrap_google_client  # type: ignore
from judgeval.tracer.keys import AttributeKeys  # type: ignore


class MockSpanProcessor:
    """Mock span processor to capture span data for testing"""

    def __init__(self):
        self.started_spans = []
        self.ended_spans = []
        self.resource_attributes = {}

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self.started_spans.append(span)

    def on_end(self, span: ReadableSpan) -> None:
        self.ended_spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def get_last_ended_span(self) -> Optional[ReadableSpan]:
        return self.ended_spans[-1] if self.ended_spans else None

    def get_span_attributes(self, span: ReadableSpan) -> dict[str, Any]:
        return dict(span.attributes or {})


class MockTracer:
    """Minimal mock tracer for testing - no API calls, just OpenTelemetry"""

    def __init__(self, tracer):
        self.tracer = tracer

    def get_tracer(self):
        return self.tracer

    def add_agent_attributes_to_span(self, span):
        """No-op for tests"""
        pass


@pytest.fixture
def mock_processor():
    return MockSpanProcessor()


@pytest.fixture
def tracer(mock_processor):
    """Minimal tracer with local OpenTelemetry only - no API, no project creation"""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import set_tracer_provider
    from judgeval.tracer.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME  # type: ignore
    from judgeval.version import get_version  # type: ignore

    provider = TracerProvider()
    provider.add_span_processor(mock_processor)
    set_tracer_provider(provider)

    otel_tracer = provider.get_tracer(
        JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME,
        get_version(),
    )

    return MockTracer(otel_tracer)


@pytest.fixture
def tracer_with_mock(tracer):
    """Alias for tracer - both now use the mock processor"""
    return tracer


@pytest.fixture
def google_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def client(google_api_key):
    return Client(api_key=google_api_key)


@pytest.fixture
def client_wrapped(tracer, client):
    """Wrapped client with tracer"""
    return wrap_google_client(tracer, client)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def client_maybe_wrapped(request, tracer, client):
    """Parametrized fixture that returns wrapped or unwrapped client"""
    if request.param == "wrapped":
        return wrap_google_client(tracer, client)
    return client


class TestWrapper:
    def test_generate_content(self, client_maybe_wrapped):
        """Test generate_content with gemini-2.5-flash"""
        response = client_maybe_wrapped.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'test' and nothing else",
        )

        assert response is not None
        assert response.text
        assert response.usage_metadata
        assert response.usage_metadata.prompt_token_count > 0
        assert response.usage_metadata.candidates_token_count > 0

    def test_multiple_calls_same_client(self, client_maybe_wrapped):
        """Test multiple calls to ensure context isolation"""
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


class TestTracingIntegration:
    def test_span_created_and_ended(self, tracer, client_maybe_wrapped):
        """Test that spans are properly created and ended"""
        response = client_maybe_wrapped.models.generate_content(
            model="gemini-2.5-flash",
            contents="Test",
        )

        assert response is not None


class TestSpanAttributes:
    def test_span_attributes_set(self, tracer, mock_processor, client_wrapped, client):
        """Test that all required span attributes are set correctly"""
        response = client_wrapped.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello",
        )

        assert response is not None

        span = mock_processor.get_last_ended_span()
        assert span is not None

        attrs = mock_processor.get_span_attributes(span)
        assert AttributeKeys.JUDGMENT_SPAN_KIND in attrs
        assert attrs[AttributeKeys.JUDGMENT_SPAN_KIND] == "llm"
        assert AttributeKeys.GEN_AI_PROMPT in attrs
        assert AttributeKeys.GEN_AI_REQUEST_MODEL in attrs
        assert attrs[AttributeKeys.GEN_AI_REQUEST_MODEL] == "gemini-2.5-flash"
        assert AttributeKeys.GEN_AI_COMPLETION in attrs
        assert AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS in attrs
        assert AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS in attrs


class TestErrorHandling:
    def test_error_recorded_in_span(self, tracer, mock_processor, client_wrapped):
        """Test that errors are properly recorded in spans"""
        with pytest.raises(Exception):
            client_wrapped.models.generate_content(
                model="invalid-model-name",
                contents="Test",
            )

        span = mock_processor.get_last_ended_span()
        assert span is not None


class TestWrapperIdempotency:
    def test_double_wrapping(self, tracer, client):
        """Test that wrapping the same client twice doesn't break anything"""
        client1 = wrap_google_client(tracer, client)
        client2 = wrap_google_client(tracer, client1)

        response = client2.models.generate_content(
            model="gemini-2.5-flash",
            contents="Test",
        )

        assert response is not None
