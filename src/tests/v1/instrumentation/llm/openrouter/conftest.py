"""OpenRouter-specific fixtures for tests."""

import pytest
import os
from typing import Any, Optional
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span

pytest.importorskip("openai")

from openai import OpenAI, AsyncOpenAI
from judgeval.v1.instrumentation.llm.llm_openai.wrapper import (
    wrap_openai_client_sync,
    wrap_openai_client_async,
)


class MockSpanProcessor:
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
    def __init__(self, tracer):
        self.tracer = tracer

    def get_tracer(self):
        return self.tracer

    def _inject_judgment_context(self, span):
        pass


@pytest.fixture
def mock_processor():
    return MockSpanProcessor()


@pytest.fixture
def tracer(mock_processor):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import set_tracer_provider
    from judgeval.tracer.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
    from judgeval.version import get_version

    provider = TracerProvider()
    provider.add_span_processor(mock_processor)
    set_tracer_provider(provider)

    otel_tracer = provider.get_tracer(
        JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME,
        get_version(),
    )

    return MockTracer(otel_tracer)


@pytest.fixture
def openrouter_api_key():
    """OpenRouter API key from environment"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def sync_client(openrouter_api_key):
    """Unwrapped sync OpenRouter client (using OpenAI SDK)"""
    return OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://judgmentlabs.ai"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Judgeval Tests"),
        },
    )


@pytest.fixture
def async_client(openrouter_api_key):
    """Unwrapped async OpenRouter client (using OpenAI SDK)"""
    return AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://judgmentlabs.ai"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Judgeval Tests"),
        },
    )


@pytest.fixture
def wrapped_sync_client(tracer, sync_client):
    """Wrapped sync OpenRouter client with tracer"""
    return wrap_openai_client_sync(tracer, sync_client)


@pytest.fixture
def wrapped_async_client(tracer, async_client):
    """Wrapped async OpenRouter client with tracer"""
    return wrap_openai_client_async(tracer, async_client)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def sync_client_maybe_wrapped(request, tracer, sync_client):
    """Parametrized fixture that yields both wrapped and unwrapped sync clients"""
    if request.param == "wrapped":
        return wrap_openai_client_sync(tracer, sync_client)
    return sync_client


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def async_client_maybe_wrapped(request, tracer, async_client):
    """Parametrized fixture that yields both wrapped and unwrapped async clients"""
    if request.param == "wrapped":
        return wrap_openai_client_async(tracer, async_client)
    return async_client
