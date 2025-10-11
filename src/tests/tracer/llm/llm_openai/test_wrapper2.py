import pytest
import os
from typing import Any, Optional

pytest.importorskip("openai")

from openai import OpenAI, AsyncOpenAI
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from judgeval.tracer.llm.llm_openai.wrapper2 import (  # type: ignore
    wrap_openai_client_sync,
    wrap_openai_client_async,
)
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

    # Set up minimal TracerProvider with mock processor
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
def openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def sync_client(openai_api_key):
    return OpenAI(api_key=openai_api_key)


@pytest.fixture
def async_client(openai_api_key):
    return AsyncOpenAI(api_key=openai_api_key)


@pytest.fixture
def wrapped_sync_client(tracer, sync_client):
    return wrap_openai_client_sync(tracer, sync_client)


@pytest.fixture
def wrapped_async_client(tracer, async_client):
    return wrap_openai_client_async(tracer, async_client)


class TestSyncWrapper:
    def test_chat_completions_create(self, wrapped_sync_client):
        """Test sync chat.completions.create with gpt-5-nano"""
        response = wrapped_sync_client.chat.completions.create(
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

    def test_responses_create(self, wrapped_sync_client):
        """Test sync responses.create with gpt-5-nano"""
        response = wrapped_sync_client.responses.create(
            model="gpt-5-nano",
            input="Say 'test' and nothing else",
        )

        assert response is not None
        assert response.model
        assert hasattr(response, "output") or hasattr(response, "text")

    def test_beta_chat_completions_parse(self, wrapped_sync_client):
        """Test sync beta.chat.completions.parse with structured outputs"""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        response = wrapped_sync_client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[
                {"role": "user", "content": "Say the word 'test' in JSON format"}
            ],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0

    def test_multiple_calls_same_client(self, wrapped_sync_client):
        """Test multiple calls to ensure context isolation"""
        response1 = wrapped_sync_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_completion_tokens=1000,
        )

        response2 = wrapped_sync_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_completion_tokens=1000,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id


class TestAsyncWrapper:
    @pytest.mark.asyncio
    async def test_chat_completions_create(self, wrapped_async_client):
        """Test async chat.completions.create with gpt-5-nano"""
        response = await wrapped_async_client.chat.completions.create(
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

    @pytest.mark.asyncio
    async def test_responses_create(self, wrapped_async_client):
        """Test async responses.create with gpt-5-nano"""
        response = await wrapped_async_client.responses.create(
            model="gpt-5-nano",
            input="Say 'test' and nothing else",
        )

        assert response is not None
        assert response.model
        assert hasattr(response, "output") or hasattr(response, "text")

    @pytest.mark.asyncio
    async def test_beta_chat_completions_parse(self, wrapped_async_client):
        """Test async beta.chat.completions.parse with structured outputs"""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        response = await wrapped_async_client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[
                {"role": "user", "content": "Say the word 'test' in JSON format"}
            ],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(self, wrapped_async_client):
        """Test multiple async calls to ensure context isolation"""
        response1 = await wrapped_async_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_completion_tokens=1000,
        )

        response2 = await wrapped_async_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_completion_tokens=1000,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id


class TestTracingIntegration:
    def test_span_created_and_ended(self, tracer, wrapped_sync_client):
        """Test that spans are properly created and ended"""
        response = wrapped_sync_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Test"}],
            max_completion_tokens=1000,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_async_span_created_and_ended(self, tracer, wrapped_async_client):
        """Test that async spans are properly created and ended"""
        response = await wrapped_async_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Test"}],
            max_completion_tokens=1000,
        )

        assert response is not None

    def test_error_handling(self, wrapped_sync_client):
        """Test that errors are properly handled and spans end"""
        with pytest.raises(Exception):
            wrapped_sync_client.chat.completions.create(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "Test"}],
            )

    @pytest.mark.asyncio
    async def test_async_error_handling(self, wrapped_async_client):
        """Test that async errors are properly handled and spans end"""
        with pytest.raises(Exception):
            await wrapped_async_client.chat.completions.create(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "Test"}],
            )


class TestSpanAttributes:
    """Test that span attributes are correctly set during tracing"""

    def test_chat_completions_span_attributes(
        self, tracer_with_mock, mock_processor, sync_client, openai_api_key
    ):
        """Test that chat.completions.create sets correct span attributes"""
        wrapped_client = wrap_openai_client_sync(tracer_with_mock, sync_client)

        response = wrapped_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_completion_tokens=1000,
        )

        assert response is not None
        assert len(mock_processor.ended_spans) > 0

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        # Verify span name
        assert span.name == "OPENAI_API_CALL"

        # Verify span kind
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

        # Verify model name
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL) == "gpt-5-nano"
        assert AttributeKeys.GEN_AI_RESPONSE_MODEL in attrs

        # Verify prompt was captured
        assert AttributeKeys.GEN_AI_PROMPT in attrs

        # Verify completion was captured
        assert AttributeKeys.GEN_AI_COMPLETION in attrs

        # Verify usage tokens
        assert AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS in attrs
        assert AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS in attrs
        assert attrs[AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS] > 0
        assert attrs[AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS] > 0

        # Verify cache tokens attribute exists
        assert AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS in attrs
        assert AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS in attrs

        # Verify usage metadata
        assert AttributeKeys.JUDGMENT_USAGE_METADATA in attrs

    @pytest.mark.asyncio
    async def test_async_chat_completions_span_attributes(
        self, tracer_with_mock, mock_processor, async_client, openai_api_key
    ):
        """Test that async chat.completions.create sets correct span attributes"""
        wrapped_client = wrap_openai_client_async(tracer_with_mock, async_client)

        response = await wrapped_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_completion_tokens=1000,
        )

        assert response is not None
        assert len(mock_processor.ended_spans) > 0

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        # Verify core attributes
        assert span.name == "OPENAI_API_CALL"
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL) == "gpt-5-nano"
        assert AttributeKeys.GEN_AI_COMPLETION in attrs
        assert attrs[AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS] > 0

    def test_responses_create_span_attributes(
        self, tracer_with_mock, mock_processor, sync_client, openai_api_key
    ):
        """Test that responses.create sets correct span attributes"""
        wrapped_client = wrap_openai_client_sync(tracer_with_mock, sync_client)

        response = wrapped_client.responses.create(
            model="gpt-5-nano",
            input="Say 'test'",
        )

        assert response is not None
        assert len(mock_processor.ended_spans) > 0

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        assert span.name == "OPENAI_API_CALL"
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL) == "gpt-5-nano"
        assert AttributeKeys.GEN_AI_PROMPT in attrs
        assert AttributeKeys.GEN_AI_COMPLETION in attrs

    def test_beta_parse_span_attributes(
        self, tracer_with_mock, mock_processor, sync_client, openai_api_key
    ):
        """Test that beta.chat.completions.parse sets correct span attributes"""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            word: str

        wrapped_client = wrap_openai_client_sync(tracer_with_mock, sync_client)

        response = wrapped_client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test' in JSON"}],
            response_format=TestResponse,
            max_completion_tokens=1000,
        )

        assert response is not None
        assert len(mock_processor.ended_spans) > 0

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        assert span.name == "OPENAI_API_CALL"
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL) == "gpt-5-nano"
        assert AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS in attrs
        assert attrs[AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS] > 0

    def test_error_span_has_exception(
        self, tracer_with_mock, mock_processor, sync_client, openai_api_key
    ):
        """Test that errors are recorded in spans"""
        wrapped_client = wrap_openai_client_sync(tracer_with_mock, sync_client)

        with pytest.raises(Exception):
            wrapped_client.chat.completions.create(
                model="invalid-model-name",
                messages=[{"role": "user", "content": "Test"}],
            )

        assert len(mock_processor.ended_spans) > 0
        span = mock_processor.get_last_ended_span()

        # Verify span exists and ended
        assert span is not None
        assert span.name == "OPENAI_API_CALL"

        # Verify span has events (exception recording)
        if span.events:
            event_names = [event.name for event in span.events]
            assert any("exception" in name.lower() for name in event_names)

    def test_multiple_spans_isolated(
        self, tracer_with_mock, mock_processor, sync_client, openai_api_key
    ):
        """Test that multiple calls create isolated spans"""
        wrapped_client = wrap_openai_client_sync(tracer_with_mock, sync_client)

        initial_span_count = len(mock_processor.ended_spans)

        wrapped_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "First"}],
            max_completion_tokens=1000,
        )

        wrapped_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Second"}],
            max_completion_tokens=1000,
        )

        # Should have 2 new spans
        assert len(mock_processor.ended_spans) == initial_span_count + 2

        # Verify spans have different contexts
        span1 = mock_processor.ended_spans[-2]
        span2 = mock_processor.ended_spans[-1]
        assert span1.context.span_id != span2.context.span_id
