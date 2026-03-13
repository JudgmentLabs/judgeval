"""Together-specific fixtures for tests."""

import pytest
import os
from typing import Any, Optional
from unittest.mock import patch

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

pytest.importorskip("together")

from together import Together, AsyncTogether
from judgeval.v1.instrumentation.llm.llm_together.wrapper import wrap_together_client
from judgeval.v1.trace.tracer import Tracer
from judgeval.v1.trace.judgment_tracer_provider import (
    JudgmentTracerProvider,
    _active_tracer_var,
)
from judgeval.v1.trace.processors.judgment_baggage_processor import (
    JudgmentBaggageProcessor,
)
from judgeval.v1.trace.exporters.noop_span_exporter import NoOpSpanExporter


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


class _BaggageAwareProcessor(SimpleSpanProcessor):
    def __init__(self, exporter, mock_processor: MockSpanProcessor):
        super().__init__(exporter)
        self._baggage = JudgmentBaggageProcessor()
        self._mock = mock_processor

    def on_start(self, span, parent_context=None):
        self._baggage.on_start(span, parent_context)
        self._mock.on_start(span, parent_context)
        super().on_start(span, parent_context)

    def on_end(self, span):
        self._mock.on_end(span)
        super().on_end(span)


@pytest.fixture(autouse=True)
def _reset_provider():
    JudgmentTracerProvider._instance = None
    _active_tracer_var.set(None)
    yield
    _active_tracer_var.set(None)
    JudgmentTracerProvider._instance = None


@pytest.fixture
def mock_processor():
    return MockSpanProcessor()


@pytest.fixture
def tracer(mock_processor):
    with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value="proj-test"):
        t = Tracer.init(
            project_name="test-together",
            api_key="test-key",
            organization_id="test-org",
            api_url="http://localhost:9999",
        )
    provider: TracerProvider = t._tracer_provider
    provider._active_span_processor._span_processors = ()  # type: ignore[attr-defined]
    provider.add_span_processor(
        _BaggageAwareProcessor(NoOpSpanExporter(), mock_processor)
    )
    return t


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


@pytest.fixture
def wrapped_sync_client(tracer, sync_client):
    return wrap_together_client(sync_client)


@pytest.fixture
def wrapped_async_client(tracer, async_client):
    return wrap_together_client(async_client)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def sync_client_maybe_wrapped(request, tracer, sync_client):
    if request.param == "wrapped":
        return wrap_together_client(sync_client)
    return sync_client


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def async_client_maybe_wrapped(request, tracer, async_client):
    if request.param == "wrapped":
        return wrap_together_client(async_client)
    return async_client
