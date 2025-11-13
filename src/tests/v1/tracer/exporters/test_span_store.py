import pytest
from unittest.mock import MagicMock
from judgeval.v1.tracer.exporters.span_store import SpanStore


@pytest.fixture
def span_store():
    return SpanStore()


@pytest.fixture
def mock_span():
    span = MagicMock()
    context = MagicMock()
    context.trace_id = 123456
    context.span_id = 789
    span.get_span_context.return_value = context
    return span


def test_span_store_initialization(span_store):
    assert span_store._spans_by_trace == {}


def test_span_store_add_span(span_store, mock_span):
    span_store.add(mock_span)
    trace_id_hex = format(123456, "032x")
    assert trace_id_hex in span_store._spans_by_trace
    assert mock_span in span_store._spans_by_trace[trace_id_hex]


def test_span_store_get_by_trace_id(span_store, mock_span):
    span_store.add(mock_span)
    trace_id_hex = format(123456, "032x")
    spans = span_store.get_by_trace_id(trace_id_hex)

    assert len(spans) == 1
    assert spans[0] == mock_span


def test_span_store_get_by_trace_id_nonexistent(span_store):
    trace_id_hex = format(999999, "032x")
    spans = span_store.get_by_trace_id(trace_id_hex)
    assert spans == []


def test_span_store_multiple_spans_same_trace(span_store):
    span1 = MagicMock()
    context1 = MagicMock()
    context1.trace_id = 123
    context1.span_id = 1
    span1.get_span_context.return_value = context1

    span2 = MagicMock()
    context2 = MagicMock()
    context2.trace_id = 123
    context2.span_id = 2
    span2.get_span_context.return_value = context2

    span_store.add(span1, span2)

    trace_id_hex = format(123, "032x")
    spans = span_store.get_by_trace_id(trace_id_hex)
    assert len(spans) == 2
    assert span1 in spans
    assert span2 in spans


def test_span_store_clear(span_store, mock_span):
    span_store.add(mock_span)
    trace_id_hex = format(123456, "032x")
    span_store.clear_trace(trace_id_hex)

    assert trace_id_hex not in span_store._spans_by_trace
