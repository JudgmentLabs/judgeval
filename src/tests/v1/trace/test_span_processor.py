from __future__ import annotations

from unittest.mock import MagicMock

from opentelemetry.trace.span import SpanContext, TraceFlags

from judgeval.v1.trace.processors.noop_judgment_span_processor import (
    NoOpJudgmentSpanProcessor,
)
from judgeval.v1.trace.processors.judgment_span_processor import JudgmentSpanProcessor
from judgeval.v1.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)


def _make_span_context(trace_id=1, span_id=2):
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


class TestNoOpJudgmentSpanProcessor:
    def setup_method(self):
        self.proc = NoOpJudgmentSpanProcessor()

    def test_on_start_does_nothing(self):
        self.proc.on_start(MagicMock())

    def test_on_end_does_nothing(self):
        self.proc.on_end(MagicMock())

    def test_force_flush_returns_true(self):
        assert self.proc.force_flush() is True

    def test_shutdown_does_nothing(self):
        self.proc.shutdown()

    def test_emit_partial_does_nothing(self):
        self.proc.emit_partial()

    def test_state_get_returns_default(self):
        ctx = _make_span_context()
        assert self.proc.state_get(ctx, "key", "default") == "default"

    def test_state_incr_returns_zero(self):
        ctx = _make_span_context()
        assert self.proc.state_incr(ctx, "counter") == 0

    def test_state_append_returns_empty(self):
        ctx = _make_span_context()
        assert self.proc.state_append(ctx, "list", "item") == []


class TestJudgmentSpanProcessorState:
    def setup_method(self):
        tracer_mock = MagicMock()
        self.proc = JudgmentSpanProcessor(tracer_mock, NoOpJudgmentSpanExporter())

    def test_state_set_and_get(self):
        ctx = _make_span_context()
        self.proc.state_set(ctx, "key", "value")
        assert self.proc.state_get(ctx, "key") == "value"

    def test_state_get_missing_returns_default(self):
        ctx = _make_span_context()
        assert self.proc.state_get(ctx, "missing", 99) == 99

    def test_state_incr_increments(self):
        ctx = _make_span_context()
        assert self.proc.state_incr(ctx, "c") == 0
        assert self.proc.state_incr(ctx, "c") == 1
        assert self.proc.state_incr(ctx, "c") == 2

    def test_state_append(self):
        ctx = _make_span_context()
        self.proc.state_append(ctx, "lst", "a")
        result = self.proc.state_append(ctx, "lst", "b")
        assert result == ["a", "b"]

    def test_state_isolated_between_spans(self):
        ctx1 = _make_span_context(trace_id=1, span_id=1)
        ctx2 = _make_span_context(trace_id=1, span_id=2)
        self.proc.state_set(ctx1, "key", "val1")
        self.proc.state_set(ctx2, "key", "val2")
        assert self.proc.state_get(ctx1, "key") == "val1"
        assert self.proc.state_get(ctx2, "key") == "val2"
