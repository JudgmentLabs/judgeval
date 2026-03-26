from __future__ import annotations

import pytest
from unittest.mock import patch

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from judgeval.trace.tracer import Tracer
from judgeval.trace.judgment_tracer_provider import (
    JudgmentTracerProvider,
    _active_tracer_var,
)
from judgeval.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)
from judgeval.trace.processors.judgment_baggage_processor import (
    JudgmentBaggageProcessor,
)


class CollectingExporter(NoOpJudgmentSpanExporter):
    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def clear(self):
        self.spans.clear()


class CollectingProcessor(SimpleSpanProcessor):
    def __init__(self, exporter):
        super().__init__(exporter)
        self._baggage = JudgmentBaggageProcessor()

    def on_start(self, span, parent_context=None):
        self._baggage.on_start(span, parent_context)
        super().on_start(span, parent_context)


@pytest.fixture(autouse=True)
def _reset_provider():
    JudgmentTracerProvider._instance = None
    _active_tracer_var.set(None)
    yield
    _active_tracer_var.set(None)
    JudgmentTracerProvider._instance = None


@pytest.fixture
def collecting_exporter():
    return CollectingExporter()


@pytest.fixture
def tracer(collecting_exporter):
    with patch("judgeval.trace.tracer.resolve_project_id", return_value="proj-123"):
        t = Tracer.init(
            project_name="test-project",
            api_key="test-key",
            organization_id="test-org",
            api_url="http://localhost:1234",
        )
    provider: TracerProvider = t._tracer_provider
    provider._active_span_processor._span_processors = ()  # type: ignore[attr-defined]
    provider.add_span_processor(CollectingProcessor(collecting_exporter))
    return t
