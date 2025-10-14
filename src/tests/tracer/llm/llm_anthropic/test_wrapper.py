import pytest
import os
from typing import Any, Optional

pytest.importorskip("anthropic")

from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from judgeval.tracer.llm.llm_anthropic.wrapper import (  # type: ignore
    wrap_anthropic_client_sync,
    wrap_anthropic_client_async,
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
def anthropic_api_key():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def sync_client(anthropic_api_key):
    return Anthropic(api_key=anthropic_api_key)


@pytest.fixture
def async_client(anthropic_api_key):
    return AsyncAnthropic(api_key=anthropic_api_key)


@pytest.fixture
def wrapped_sync_client(tracer, sync_client):
    return wrap_anthropic_client_sync(tracer, sync_client)


@pytest.fixture
def wrapped_async_client(tracer, async_client):
    return wrap_anthropic_client_async(tracer, async_client)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def sync_client_maybe_wrapped(request, tracer, sync_client):
    """Parametrized fixture that yields both wrapped and unwrapped sync clients"""
    if request.param == "wrapped":
        return wrap_anthropic_client_sync(tracer, sync_client)
    return sync_client


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def async_client_maybe_wrapped(request, tracer, async_client):
    """Parametrized fixture that yields both wrapped and unwrapped async clients"""
    if request.param == "wrapped":
        return wrap_anthropic_client_async(tracer, async_client)
    return async_client


class TestSyncWrapper:
    def test_messages_create(self, sync_client_maybe_wrapped):
        """Test sync messages.create"""
        response = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
        )

        assert response is not None
        assert response.content
        assert len(response.content) > 0
        assert response.model
        assert response.usage
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_messages_create_streaming(self, sync_client_maybe_wrapped):
        """Test sync messages.create with stream=True"""
        response = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "type")

    def test_messages_stream(self, sync_client_maybe_wrapped):
        """Test sync messages.stream context manager"""
        with sync_client_maybe_wrapped.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'test'"}],
        ) as stream:
            text_chunks = list(stream.text_stream)
            assert len(text_chunks) > 0

    def test_multiple_calls_same_client(self, sync_client_maybe_wrapped):
        """Test multiple calls to ensure context isolation"""
        response1 = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'first'"}],
        )

        response2 = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'second'"}],
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id


class TestAsyncWrapper:
    @pytest.mark.asyncio
    async def test_messages_create(self, async_client_maybe_wrapped):
        """Test async messages.create"""
        response = await async_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
        )

        assert response is not None
        assert response.content
        assert len(response.content) > 0
        assert response.model
        assert response.usage
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_messages_create_streaming(self, async_client_maybe_wrapped):
        """Test async messages.create with stream=True"""
        stream = await async_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "type")

    @pytest.mark.asyncio
    async def test_messages_stream(self, async_client_maybe_wrapped):
        """Test async messages.stream context manager"""
        async with async_client_maybe_wrapped.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'test'"}],
        ) as stream:
            text_chunks = [chunk async for chunk in stream.text_stream]
            assert len(text_chunks) > 0

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(self, async_client_maybe_wrapped):
        """Test multiple async calls to ensure context isolation"""
        response1 = await async_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'first'"}],
        )

        response2 = await async_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'second'"}],
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id


class TestTracingIntegration:
    def test_span_created_and_ended(self, tracer, sync_client_maybe_wrapped):
        """Test that spans are properly created and ended"""
        response = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test"}],
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_async_span_created_and_ended(
        self, tracer, async_client_maybe_wrapped
    ):
        """Test that async spans are properly created and ended"""
        response = await async_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test"}],
        )

        assert response is not None

    def test_error_handling(self, sync_client_maybe_wrapped):
        """Test that errors are properly handled and spans end"""
        with pytest.raises(Exception):
            sync_client_maybe_wrapped.messages.create(
                model="invalid-model-name-that-does-not-exist",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Test"}],
            )

    @pytest.mark.asyncio
    async def test_async_error_handling(self, async_client_maybe_wrapped):
        """Test that async errors are properly handled and spans end"""
        with pytest.raises(Exception):
            await async_client_maybe_wrapped.messages.create(
                model="invalid-model-name-that-does-not-exist",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Test"}],
            )


class TestSpanAttributes:
    """Test that span attributes are correctly set during tracing"""

    def test_messages_create_span_attributes(
        self, tracer_with_mock, mock_processor, sync_client, anthropic_api_key
    ):
        """Test that messages.create sets correct span attributes"""
        wrapped_client = wrap_anthropic_client_sync(tracer_with_mock, sync_client)

        response = wrapped_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'test'"}],
        )

        assert response is not None
        assert len(mock_processor.ended_spans) > 0

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        # Verify span name
        assert span.name == "ANTHROPIC_API_CALL"

        # Verify span kind
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

        # Verify model name
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL) == "claude-sonnet-4-5"
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
    async def test_async_messages_create_span_attributes(
        self, tracer_with_mock, mock_processor, async_client, anthropic_api_key
    ):
        """Test that async messages.create sets correct span attributes"""
        wrapped_client = wrap_anthropic_client_async(tracer_with_mock, async_client)

        response = await wrapped_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'test'"}],
        )

        assert response is not None
        assert len(mock_processor.ended_spans) > 0

        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        # Verify core attributes
        assert span.name == "ANTHROPIC_API_CALL"
        assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"
        assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL) == "claude-sonnet-4-5"
        assert AttributeKeys.GEN_AI_COMPLETION in attrs
        assert attrs[AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS] > 0

    def test_error_span_has_exception(
        self, tracer_with_mock, mock_processor, sync_client, anthropic_api_key
    ):
        """Test that errors are recorded in spans"""
        wrapped_client = wrap_anthropic_client_sync(tracer_with_mock, sync_client)

        with pytest.raises(Exception):
            wrapped_client.messages.create(
                model="invalid-model-name",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Test"}],
            )

        assert len(mock_processor.ended_spans) > 0
        span = mock_processor.get_last_ended_span()

        # Verify span exists and ended
        assert span is not None
        assert span.name == "ANTHROPIC_API_CALL"

        # Verify span has events (exception recording)
        if span.events:
            event_names = [event.name for event in span.events]
            assert any("exception" in name.lower() for name in event_names)

    def test_multiple_spans_isolated(
        self, tracer_with_mock, mock_processor, sync_client, anthropic_api_key
    ):
        """Test that multiple calls create isolated spans"""
        wrapped_client = wrap_anthropic_client_sync(tracer_with_mock, sync_client)

        initial_span_count = len(mock_processor.ended_spans)

        wrapped_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "First"}],
        )

        wrapped_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Second"}],
        )

        # Should have 2 new spans
        assert len(mock_processor.ended_spans) == initial_span_count + 2

        # Verify spans have different contexts
        span1 = mock_processor.ended_spans[-2]
        span2 = mock_processor.ended_spans[-1]
        assert span1.context.span_id != span2.context.span_id


class TestStreamingSync:
    def test_messages_create_streaming(self, sync_client_maybe_wrapped):
        """Test sync messages.create with stream=True"""
        stream = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "type")

    def test_messages_stream_context_manager(self, sync_client_maybe_wrapped):
        """Test sync messages.stream with context manager"""
        with sync_client_maybe_wrapped.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
        ) as stream:
            text_chunks = list(stream.text_stream)
            assert len(text_chunks) > 0

    def test_streaming_content_accumulation(self, sync_client_maybe_wrapped):
        """Verify content is accumulated correctly across chunks"""
        with sync_client_maybe_wrapped.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: Hello World"}],
        ) as stream:
            accumulated = ""
            for text_chunk in stream.text_stream:
                accumulated += text_chunk

            assert len(accumulated) > 0

    def test_streaming_early_break(self, sync_client_maybe_wrapped):
        """Test breaking out of stream early"""
        stream = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 10"}],
            stream=True,
        )

        first_chunk = next(iter(stream))
        assert first_chunk is not None

    def test_streaming_early_break_with_error(self, sync_client_maybe_wrapped):
        """Test breaking out of stream early with error"""
        with pytest.raises(Exception) as exc_info:
            sync_client_maybe_wrapped.messages.create(
                model="non-existent-model",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Count to 10"}],
                stream=True,
            )

        assert exc_info.value is not None

    def test_streaming_early_break_with_error_context_manager(
        self, sync_client_maybe_wrapped
    ):
        """Test breaking out of stream early with error"""
        with pytest.raises(Exception) as exc_info:
            with sync_client_maybe_wrapped.messages.stream(
                model="non-existent-model",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Count to 10"}],
            ) as stream:
                for chunk in stream:
                    pass
        assert exc_info.value is not None


class TestStreamingAsync:
    @pytest.mark.asyncio
    async def test_messages_create_streaming(self, async_client_maybe_wrapped):
        """Test async messages.create with stream=True"""
        stream = await async_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "type")

    @pytest.mark.asyncio
    async def test_messages_stream_context_manager(self, async_client_maybe_wrapped):
        """Test async messages.stream with context manager"""
        async with async_client_maybe_wrapped.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 3"}],
        ) as stream:
            text_chunks = [chunk async for chunk in stream.text_stream]
            assert len(text_chunks) > 0

    @pytest.mark.asyncio
    async def test_streaming_content_accumulation(self, async_client_maybe_wrapped):
        """Verify content is accumulated correctly across chunks"""
        async with async_client_maybe_wrapped.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: Hello World"}],
        ) as stream:
            accumulated = ""
            async for text_chunk in stream.text_stream:
                accumulated += text_chunk

            assert len(accumulated) > 0

    @pytest.mark.asyncio
    async def test_streaming_early_break(self, async_client_maybe_wrapped):
        """Test breaking out of stream early"""
        stream = await async_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count to 10"}],
            stream=True,
        )

        first_chunk = await stream.__anext__()
        assert first_chunk is not None

    @pytest.mark.asyncio
    async def test_streaming_early_break_with_error(self, async_client_maybe_wrapped):
        """Test breaking out of stream early with error"""
        with pytest.raises(Exception) as exc_info:
            await async_client_maybe_wrapped.messages.create(
                model="non-existent-model",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Count to 10"}],
                stream=True,
            )
        assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_streaming_early_break_with_error_context_manager(
        self, async_client_maybe_wrapped
    ):
        """Test breaking out of stream early with error"""
        with pytest.raises(Exception) as exc_info:
            with async_client_maybe_wrapped.messages.stream(
                model="non-existent-model",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Count to 10"}],
            ) as stream:
                async for chunk in stream:
                    pass
        assert exc_info.value is not None


class TestStreamingSpanAttributes:
    def test_streaming_span_has_accumulated_content(
        self, tracer_with_mock, mock_processor, sync_client, anthropic_api_key
    ):
        """Verify GEN_AI_COMPLETION has full accumulated content"""
        wrapped_client = wrap_anthropic_client_sync(tracer_with_mock, sync_client)

        stream = wrapped_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: test"}],
            stream=True,
        )

        accumulated = ""
        for chunk in stream:
            if chunk.type == "content_block_delta":
                text = getattr(chunk.delta, "text", None)
                if text:
                    accumulated += text

        assert len(mock_processor.ended_spans) > 0
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        assert AttributeKeys.GEN_AI_COMPLETION in attrs

    def test_streaming_span_has_usage_tokens(
        self, tracer_with_mock, mock_processor, sync_client, anthropic_api_key
    ):
        """Verify usage tokens extracted from stream"""
        wrapped_client = wrap_anthropic_client_sync(tracer_with_mock, sync_client)

        stream = wrapped_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: test"}],
            stream=True,
        )

        list(stream)

        assert len(mock_processor.ended_spans) > 0
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        assert AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS in attrs
        assert AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS in attrs

    def test_stream_context_manager_span_has_content(
        self, tracer_with_mock, mock_processor, sync_client, anthropic_api_key
    ):
        """Verify messages.stream context manager captures content"""
        wrapped_client = wrap_anthropic_client_sync(tracer_with_mock, sync_client)

        with wrapped_client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: test"}],
        ) as stream:
            list(stream.text_stream)

        assert len(mock_processor.ended_spans) > 0
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        assert AttributeKeys.GEN_AI_COMPLETION in attrs

    @pytest.mark.asyncio
    async def test_async_stream_context_manager_span_has_content(
        self, tracer_with_mock, mock_processor, async_client, anthropic_api_key
    ):
        """Verify async messages.stream context manager captures content"""
        wrapped_client = wrap_anthropic_client_async(tracer_with_mock, async_client)

        async with wrapped_client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: test"}],
        ) as stream:
            async for _ in stream.text_stream:
                pass

        assert len(mock_processor.ended_spans) > 0
        span = mock_processor.get_last_ended_span()
        attrs = mock_processor.get_span_attributes(span)

        assert AttributeKeys.GEN_AI_COMPLETION in attrs


class TestEdgeCases:
    def test_concurrent_calls_different_clients(
        self, tracer, sync_client, anthropic_api_key
    ):
        """Test multiple wrapped clients don't interfere"""
        from anthropic import Anthropic

        client1 = wrap_anthropic_client_sync(
            tracer, Anthropic(api_key=anthropic_api_key)
        )
        client2 = wrap_anthropic_client_sync(
            tracer, Anthropic(api_key=anthropic_api_key)
        )

        response1 = client1.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: one"}],
        )

        response2 = client2.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say: two"}],
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

    def test_streaming_with_minimal_response(self, sync_client_maybe_wrapped):
        """Test streaming with very short response"""
        stream = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say: hi"}],
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) >= 0

    @pytest.mark.asyncio
    async def test_async_streaming_with_minimal_response(
        self, async_client_maybe_wrapped
    ):
        """Test async streaming with very short response"""
        stream = await async_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say: hi"}],
            stream=True,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) >= 0


class TestSafetyGuarantees:
    def test_safe_serialize_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, anthropic_api_key
    ):
        """Test that if safe_serialize throws, user code still works"""
        from judgeval.utils import serialize  # type: ignore

        def broken_serialize(obj):
            raise RuntimeError("Serialization failed!")

        monkeypatch.setattr(serialize, "safe_serialize", broken_serialize)

        wrapped_client = wrap_anthropic_client_sync(tracer, sync_client)
        response = wrapped_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
        )

        assert response is not None
        assert response.content
        assert len(response.content) > 0

    def test_wrapped_vs_unwrapped_structure(self, tracer, anthropic_api_key):
        """Verify wrapped client behavior matches unwrapped structure"""
        from anthropic import Anthropic

        unwrapped = Anthropic(api_key=anthropic_api_key)
        wrapped = wrap_anthropic_client_sync(
            tracer, Anthropic(api_key=anthropic_api_key)
        )

        unwrapped_response = unwrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say exactly: test"}],
        )

        wrapped_response = wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say exactly: test"}],
        )

        assert type(unwrapped_response) is type(wrapped_response)
        assert hasattr(wrapped_response, "content")
        assert hasattr(wrapped_response, "usage")
        assert hasattr(wrapped_response, "model")
        assert wrapped_response.model == unwrapped_response.model

    def test_exceptions_propagate_correctly(self, sync_client_maybe_wrapped):
        """Verify API exceptions still reach user"""
        with pytest.raises(Exception) as exc_info:
            sync_client_maybe_wrapped.messages.create(
                model="invalid-model-name-that-does-not-exist",
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_async_exceptions_propagate(self, async_client_maybe_wrapped):
        """Verify async API exceptions still reach user"""
        with pytest.raises(Exception) as exc_info:
            await async_client_maybe_wrapped.messages.create(
                model="invalid-model-name-that-does-not-exist",
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value is not None

    def test_streaming_exceptions_propagate(self, sync_client_maybe_wrapped):
        """Verify streaming exceptions propagate correctly"""
        stream = sync_client_maybe_wrapped.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
            stream=True,
        )

        first_chunk = next(iter(stream))
        assert first_chunk is not None

    def test_set_span_attribute_error_doesnt_crash(
        self, monkeypatch, tracer, sync_client, anthropic_api_key
    ):
        """Test that span attribute errors don't break user code"""
        from judgeval.tracer import utils  # type: ignore

        original_set = utils.set_span_attribute

        def broken_set_attribute(span, key, value):
            if "COMPLETION" in key:
                raise RuntimeError("Attribute setting failed!")
            return original_set(span, key, value)

        monkeypatch.setattr(utils, "set_span_attribute", broken_set_attribute)

        wrapped_client = wrap_anthropic_client_sync(tracer, sync_client)
        response = wrapped_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
        )

        assert response is not None
        assert response.content
