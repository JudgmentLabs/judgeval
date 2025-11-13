"""Shared fixtures for all LLM wrapper tests."""

import pytest
from typing import Any, Optional
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span


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

    def _inject_judgment_context(self, span):
        """No-op for tests"""
        pass


@pytest.fixture
def mock_processor():
    """Mock span processor for capturing spans"""
    return MockSpanProcessor()


@pytest.fixture
def tracer(mock_processor):
    """Minimal tracer with local OpenTelemetry only - no API, no project creation"""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import (
        set_tracer_provider,
        _TRACER_PROVIDER_SET_ONCE,
        _TRACER_PROVIDER,
    )
    from judgeval.tracer.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
    from judgeval.tracer import Tracer
    from judgeval.utils.meta import SingletonMeta
    from judgeval.version import get_version

    # Clear any existing Tracer singleton
    if Tracer in SingletonMeta._instances:
        del SingletonMeta._instances[Tracer]

    # Reset the global tracer provider flag (OpenTelemetry internal)
    try:
        _TRACER_PROVIDER_SET_ONCE._done = False
        _TRACER_PROVIDER._default = None
    except Exception:
        pass  # If the internal API changes, just continue

    # Set up minimal TracerProvider with mock processor
    provider = TracerProvider()
    provider.add_span_processor(mock_processor)
    set_tracer_provider(provider)

    otel_tracer = provider.get_tracer(
        JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME,
        get_version(),
    )

    yield MockTracer(otel_tracer)

    # Cleanup after test
    mock_processor.force_flush()
    if Tracer in SingletonMeta._instances:
        del SingletonMeta._instances[Tracer]


@pytest.fixture
def tracer_with_mock(tracer):
    """Alias for tracer - both now use the mock processor"""
    return tracer
