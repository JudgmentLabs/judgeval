"""Tests for JudgmentTracerProvider and ProxyTracer."""

from __future__ import annotations

from unittest.mock import MagicMock


from judgeval.v1.trace.judgment_tracer_provider import JudgmentTracerProvider


class TestSingleton:
    def test_returns_same_instance(self):
        a = JudgmentTracerProvider.get_instance()
        b = JudgmentTracerProvider.get_instance()
        assert a is b

    def test_reset_clears_instance(self):
        a = JudgmentTracerProvider.get_instance()
        JudgmentTracerProvider._instance = None
        b = JudgmentTracerProvider.get_instance()
        assert a is not b


class TestTracerRegistration:
    def test_register_and_set_active(self, tracer):
        provider = JudgmentTracerProvider.get_instance()
        assert provider.get_active_tracer() is tracer

    def test_deregister_removes_tracer(self, tracer):
        provider = JudgmentTracerProvider.get_instance()
        provider.deregister(tracer)
        assert tracer not in provider._tracers

    def test_set_active_blocked_during_root_span(self, tracer, collecting_exporter):
        """Cannot swap the active tracer while a root span is recording."""
        from judgeval.v1.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("root"):
            mock_tracer = MagicMock()
            provider = JudgmentTracerProvider.get_instance()
            assert provider.set_active(mock_tracer) is False
            # Original tracer still active
            assert provider.get_active_tracer() is tracer


class TestSpanContext:
    def test_get_current_span_returns_invalid_when_no_span(self):
        provider = JudgmentTracerProvider.get_instance()
        span = provider.get_current_span()
        assert not span.is_recording()

    def test_has_active_root_span(self, tracer):
        from judgeval.v1.trace.base_tracer import BaseTracer

        provider = JudgmentTracerProvider.get_instance()
        assert provider.has_active_root_span() is False

        with BaseTracer.start_as_current_span("root"):
            assert provider.has_active_root_span() is True

        assert provider.has_active_root_span() is False


class TestContextManagement:
    def test_use_span_records_exception(self, tracer, collecting_exporter):
        from judgeval.v1.trace.base_tracer import BaseTracer
        import pytest

        with pytest.raises(ValueError, match="boom"):
            with BaseTracer.start_as_current_span("failing"):
                raise ValueError("boom")

        assert len(collecting_exporter.spans) >= 1
        span = collecting_exporter.spans[-1]
        assert span.status.status_code.name == "ERROR"
        assert len(span.events) >= 1
        assert span.events[0].name == "exception"


class TestProxyTracer:
    def test_proxy_delegates_to_active_provider(self, tracer, collecting_exporter):
        """ProxyTracer should create real spans when a tracer is active."""
        from judgeval.v1.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("test-span") as span:
            assert span.is_recording()
            span.set_attribute("key", "value")

        assert any(s.name == "test-span" for s in collecting_exporter.spans)

    def test_proxy_returns_noop_when_no_active_tracer(self):
        """Without an active tracer, ProxyTracer yields non-recording spans."""
        provider = JudgmentTracerProvider.get_instance()
        proxy = provider._proxy_tracer
        span = proxy.start_span("should-be-noop")
        assert not span.is_recording()
