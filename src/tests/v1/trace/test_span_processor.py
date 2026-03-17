"""Tests for JudgmentSpanProcessor and NoOpSpanProcessor."""

from __future__ import annotations

from unittest.mock import MagicMock

from opentelemetry.trace.span import SpanContext, TraceFlags

from judgeval.v1.trace.processors.judgment_span_processor import JudgmentSpanProcessor
from judgeval.v1.trace.processors.noop_judgment_span_processor import (
    NoOpJudgmentSpanProcessor,
)
from judgeval.v1.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)


def _make_span_context(trace_id=1, span_id=2):
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(1),
    )


class TestSpanProcessorState:
    def _make_processor(self):
        mock_tracer = MagicMock()
        return JudgmentSpanProcessor(mock_tracer, NoOpJudgmentSpanExporter())

    def test_state_set_and_get(self):
        proc = self._make_processor()
        ctx = _make_span_context()
        proc.state_set(ctx, "key", "value")
        assert proc.state_get(ctx, "key") == "value"

    def test_state_get_default(self):
        proc = self._make_processor()
        ctx = _make_span_context()
        assert proc.state_get(ctx, "missing") is None
        assert proc.state_get(ctx, "missing", 42) == 42

    def test_state_incr(self):
        proc = self._make_processor()
        ctx = _make_span_context()

        assert proc.state_incr(ctx, "counter") == 0  # returns value before increment
        assert proc.state_incr(ctx, "counter") == 1
        assert proc.state_incr(ctx, "counter") == 2

    def test_state_append(self):
        proc = self._make_processor()
        ctx = _make_span_context()

        result = proc.state_append(ctx, "items", "a")
        assert result == ["a"]
        result = proc.state_append(ctx, "items", "b")
        assert result == ["a", "b"]

    def test_state_isolated_between_spans(self):
        proc = self._make_processor()
        ctx1 = _make_span_context(trace_id=1, span_id=1)
        ctx2 = _make_span_context(trace_id=1, span_id=2)

        proc.state_set(ctx1, "key", "span1")
        proc.state_set(ctx2, "key", "span2")
        assert proc.state_get(ctx1, "key") == "span1"
        assert proc.state_get(ctx2, "key") == "span2"

    def test_cleanup_removes_state(self):
        proc = self._make_processor()
        ctx = _make_span_context(trace_id=10, span_id=20)
        proc.state_set(ctx, "key", "val")
        proc._cleanup_span_state((10, 20))
        assert proc.state_get(ctx, "key") is None


class TestNoOpJudgmentSpanProcessor:
    def test_noop_methods_dont_raise(self):
        proc = NoOpJudgmentSpanProcessor()
        proc.on_start(MagicMock(), None)
        proc.on_end(MagicMock())
        proc.shutdown()
        assert proc.force_flush() is True
        proc.emit_partial()

    def test_noop_state_get_returns_default(self):
        proc = NoOpJudgmentSpanProcessor()
        ctx = _make_span_context()
        assert proc.state_get(ctx, "key") is None
        assert proc.state_get(ctx, "key", "default") == "default"

    def test_noop_state_incr_returns_zero(self):
        proc = NoOpJudgmentSpanProcessor()
        ctx = _make_span_context()
        assert proc.state_incr(ctx, "counter") == 0
        assert proc.state_incr(ctx, "counter") == 0

    def test_noop_state_append_returns_empty(self):
        proc = NoOpJudgmentSpanProcessor()
        ctx = _make_span_context()
        assert proc.state_append(ctx, "items", "a") == []
