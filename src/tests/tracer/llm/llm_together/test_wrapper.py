import pytest
import os
from typing import Any, Optional

pytest.importorskip("together")

from together import Together, AsyncTogether  # type: ignore[import-untyped]
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from judgeval.tracer.llm.llm_together.wrapper import (
    wrap_together_client,
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
def together_api_key():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        pytest.skip("TOGETHER_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def sync_client(together_api_key):
    return Together(api_key=together_api_key)


@pytest.fixture
def async_client(together_api_key):
    return AsyncTogether(api_key=together_api_key)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def sync_client_maybe_wrapped(request, tracer, sync_client):
    """Parametrized fixture that yields both wrapped and unwrapped sync clients"""
    if request.param == "wrapped":
        return wrap_together_client(tracer, sync_client)
    return sync_client


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def async_client_maybe_wrapped(request, tracer, async_client):
    """Parametrized fixture that yields both wrapped and unwrapped async clients"""
    if request.param == "wrapped":
        return wrap_together_client(tracer, async_client)
    return async_client


class TestSyncWrapper:
    def test_chat_completions_create(self, sync_client_maybe_wrapped):
        """Test sync chat.completions.create"""
        response = sync_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_tokens=100,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_chat_completions_create_streaming(self, sync_client_maybe_wrapped):
        """Test sync streaming chat.completions.create"""
        stream = sync_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0

        # Check that we got content in at least one chunk
        content_chunks = [
            chunk
            for chunk in chunks
            if chunk.choices and chunk.choices[0].delta.content
        ]
        assert len(content_chunks) > 0

        # Check last chunk has usage if available
        if chunks[-1].usage:
            assert chunks[-1].usage.prompt_tokens > 0

    def test_multiple_calls_same_client(self, sync_client_maybe_wrapped):
        """Test multiple calls to ensure context isolation"""
        response1 = sync_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_tokens=50,
        )

        response2 = sync_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_tokens=50,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id


class TestAsyncWrapper:
    @pytest.mark.asyncio
    async def test_chat_completions_create(self, async_client_maybe_wrapped):
        """Test async chat.completions.create"""
        response = await async_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_tokens=100,
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
    async def test_chat_completions_create_streaming(self, async_client_maybe_wrapped):
        """Test async streaming chat.completions.create"""
        stream = await async_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

        # Check that we got content in at least one chunk
        content_chunks = [
            chunk
            for chunk in chunks
            if chunk.choices and chunk.choices[0].delta.content
        ]
        assert len(content_chunks) > 0

        # Check last chunk has usage if available
        if chunks[-1].usage:
            assert chunks[-1].usage.prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(self, async_client_maybe_wrapped):
        """Test multiple async calls to ensure context isolation"""
        response1 = await async_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_tokens=50,
        )

        response2 = await async_client_maybe_wrapped.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_tokens=50,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id


class TestTracingAttributes:
    """Test that tracing attributes are correctly set"""

    def test_sync_non_streaming_span_attributes(
        self, tracer, sync_client, mock_processor
    ):
        """Test span attributes for sync non-streaming call"""
        wrapped_client = wrap_together_client(tracer, sync_client)

        response = wrapped_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

        assert response is not None

        span = mock_processor.get_last_ended_span()
        assert span is not None

        attrs = mock_processor.get_span_attributes(span)
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL).startswith("together_ai/")
        assert AttributeKeys.GEN_AI_PROMPT in attrs
        assert AttributeKeys.GEN_AI_COMPLETION in attrs
        assert attrs.get(AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, 0) > 0
        assert attrs.get(AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, 0) > 0

    def test_sync_streaming_span_attributes(self, tracer, sync_client, mock_processor):
        """Test span attributes for sync streaming call"""
        wrapped_client = wrap_together_client(tracer, sync_client)

        stream = wrapped_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
            stream=True,
        )

        # Consume the stream
        list(stream)

        span = mock_processor.get_last_ended_span()
        assert span is not None

        attrs = mock_processor.get_span_attributes(span)
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL).startswith("together_ai/")
        assert AttributeKeys.GEN_AI_PROMPT in attrs
        assert AttributeKeys.GEN_AI_COMPLETION in attrs

    @pytest.mark.asyncio
    async def test_async_non_streaming_span_attributes(
        self, tracer, async_client, mock_processor
    ):
        """Test span attributes for async non-streaming call"""
        wrapped_client = wrap_together_client(tracer, async_client)

        response = await wrapped_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

        assert response is not None

        span = mock_processor.get_last_ended_span()
        assert span is not None

        attrs = mock_processor.get_span_attributes(span)
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL).startswith("together_ai/")
        assert AttributeKeys.GEN_AI_PROMPT in attrs
        assert AttributeKeys.GEN_AI_COMPLETION in attrs
        assert attrs.get(AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, 0) > 0
        assert attrs.get(AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, 0) > 0

    @pytest.mark.asyncio
    async def test_async_streaming_span_attributes(
        self, tracer, async_client, mock_processor
    ):
        """Test span attributes for async streaming call"""
        wrapped_client = wrap_together_client(tracer, async_client)

        stream = await wrapped_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
            stream=True,
        )

        # Consume the stream
        async for _ in stream:
            pass

        span = mock_processor.get_last_ended_span()
        assert span is not None

        attrs = mock_processor.get_span_attributes(span)
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL).startswith("together_ai/")
        assert AttributeKeys.GEN_AI_PROMPT in attrs
        assert AttributeKeys.GEN_AI_COMPLETION in attrs


class TestIdempotency:
    """Test that wrapping is idempotent and doesn't affect unwrapped clients"""

    def test_double_wrap_sync(self, tracer, sync_client):
        """Test that double wrapping doesn't break the client"""
        wrapped_once = wrap_together_client(tracer, sync_client)
        wrapped_twice = wrap_together_client(tracer, wrapped_once)

        response = wrapped_twice.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

        assert response is not None
        assert response.choices

    @pytest.mark.asyncio
    async def test_double_wrap_async(self, tracer, async_client):
        """Test that double wrapping async client doesn't break it"""
        wrapped_once = wrap_together_client(tracer, async_client)
        wrapped_twice = wrap_together_client(tracer, wrapped_once)

        response = await wrapped_twice.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

        assert response is not None
        assert response.choices
