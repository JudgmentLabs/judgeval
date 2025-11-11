import pytest
from unittest.mock import MagicMock
from opentelemetry.sdk.trace.export import SpanExportResult
from judgeval.v1.tracer.exporters.in_memory_span_exporter import InMemorySpanExporter
from judgeval.v1.tracer.exporters.span_store import SpanStore


@pytest.fixture
def span_store():
    return SpanStore()


@pytest.fixture
def exporter(span_store):
    return InMemorySpanExporter(span_store)


@pytest.fixture
def mock_span():
    span = MagicMock()
    context = MagicMock()
    context.trace_id = 123456
    span.get_span_context.return_value = context
    return span


def test_exporter_initialization(exporter, span_store):
    assert exporter._store == span_store


def test_exporter_export(exporter, span_store, mock_span):
    result = exporter.export([mock_span])

    assert result == SpanExportResult.SUCCESS
    trace_id_hex = format(123456, "032x")
    spans = span_store.get_by_trace_id(trace_id_hex)
    assert len(spans) == 1
    assert spans[0] == mock_span


def test_exporter_export_multiple_spans(exporter, span_store):
    span1 = MagicMock()
    context1 = MagicMock()
    context1.trace_id = 123
    span1.get_span_context.return_value = context1

    span2 = MagicMock()
    context2 = MagicMock()
    context2.trace_id = 123
    span2.get_span_context.return_value = context2

    result = exporter.export([span1, span2])

    assert result == SpanExportResult.SUCCESS
    trace_id_hex = format(123, "032x")
    spans = span_store.get_by_trace_id(trace_id_hex)
    assert len(spans) == 2


def test_exporter_shutdown(exporter):
    result = exporter.shutdown()
    assert result is None
