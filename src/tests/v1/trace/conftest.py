from __future__ import annotations

import pytest
from unittest.mock import patch

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from judgeval.v1.trace.tracer import Tracer
from judgeval.v1.trace.judgment_tracer_provider import (
    JudgmentTracerProvider,
    _active_tracer_var,
)
from judgeval.v1.trace.exporters.noop_span_exporter import NoOpSpanExporter
from judgeval.v1.trace.processors.judgment_baggage_processor import (
    JudgmentBaggageProcessor,
)


class CollectingExporter(NoOpSpanExporter):
    """Captures exported spans in memory for assertion."""

    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def clear(self):
        self.spans.clear()


class CollectingProcessor(SimpleSpanProcessor):
    """SimpleSpanProcessor that also runs JudgmentBaggageProcessor.on_start
    so baggage attributes propagate to child spans in tests."""

    def __init__(self, exporter):
        super().__init__(exporter)
        self._baggage = JudgmentBaggageProcessor()

    def on_start(self, span, parent_context=None):
        self._baggage.on_start(span, parent_context)
        super().on_start(span, parent_context)


@pytest.fixture(autouse=True)
def _reset_provider():
    """Reset the singleton JudgmentTracerProvider between tests."""
    JudgmentTracerProvider._instance = None
    _token = _active_tracer_var.set(None)
    yield
    _active_tracer_var.set(None)
    JudgmentTracerProvider._instance = None


@pytest.fixture
def collecting_exporter():
    return CollectingExporter()


@pytest.fixture
def tracer(collecting_exporter):
    """Create a real Tracer wired to a CollectingExporter (no network).

    Patches resolve_project_id so no API call is made.
    """
    with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value="proj-123"):
        t = Tracer.init(
            project_name="test-project",
            api_key="test-key",
            organization_id="test-org",
            api_url="http://localhost:1234",
        )
    # Replace the real exporter/processor with our collector
    provider: TracerProvider = t._tracer_provider
    provider._active_span_processor._span_processors = ()  # type: ignore[attr-defined]
    provider.add_span_processor(CollectingProcessor(collecting_exporter))
    return t
