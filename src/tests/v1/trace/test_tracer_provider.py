from __future__ import annotations

from unittest.mock import patch

from opentelemetry.trace import NoOpTracer

from judgeval.v1.trace.tracer import Tracer
from judgeval.v1.trace.judgment_tracer_provider import JudgmentTracerProvider


def _make_tracer(**kwargs):
    defaults = dict(
        project_name="p", api_key="k", organization_id="o", api_url="http://x"
    )
    defaults.update(kwargs)
    with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value="pid"):
        return Tracer.init(**defaults)


class TestSingleton:
    def test_same_instance(self):
        a = JudgmentTracerProvider.get_instance()
        b = JudgmentTracerProvider.get_instance()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = JudgmentTracerProvider.get_instance()
        JudgmentTracerProvider._instance = None
        b = JudgmentTracerProvider.get_instance()
        assert a is not b


class TestTracerRegistration:
    def test_register_and_get_active(self):
        t = _make_tracer()
        provider = JudgmentTracerProvider.get_instance()
        assert provider.get_active_tracer() is t

    def test_deregister(self):
        t = _make_tracer()
        provider = JudgmentTracerProvider.get_instance()
        provider.deregister(t)
        assert t not in provider._tracers

    def test_set_active_returns_true(self):
        t = _make_tracer()
        provider = JudgmentTracerProvider.get_instance()
        result = provider.set_active(t)
        assert result is True

    def test_set_active_blocked_during_root_span(self, tracer, collecting_exporter):
        from judgeval.v1.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("root"):
            t2 = _make_tracer(set_active=False)
            provider = JudgmentTracerProvider.get_instance()
            result = provider.set_active(t2)
            assert result is False


class TestGetCurrentSpan:
    def test_no_span_returns_invalid(self, tracer):
        from opentelemetry.trace import INVALID_SPAN

        provider = JudgmentTracerProvider.get_instance()
        assert provider.get_current_span() is INVALID_SPAN

    def test_span_available_inside_context(self, tracer):
        from judgeval.v1.trace.base_tracer import BaseTracer

        provider = JudgmentTracerProvider.get_instance()
        with BaseTracer.start_as_current_span("s"):
            span = provider.get_current_span()
            assert span.is_recording()


class TestHasActiveRootSpan:
    def test_false_when_no_span(self, tracer):
        assert JudgmentTracerProvider.get_instance().has_active_root_span() is False

    def test_true_at_root(self, tracer):
        from judgeval.v1.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("root"):
            assert JudgmentTracerProvider.get_instance().has_active_root_span() is True

    def test_false_for_child_span(self, tracer):
        from judgeval.v1.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("root"):
            with BaseTracer.start_as_current_span("child"):
                assert (
                    JudgmentTracerProvider.get_instance().has_active_root_span()
                    is False
                )


class TestGetTracer:
    def test_returns_proxy_tracer(self):
        from judgeval.v1.trace.judgment_tracer_provider import ProxyTracer

        provider = JudgmentTracerProvider.get_instance()
        t = provider.get_tracer("some-lib")
        assert isinstance(t, ProxyTracer)


class TestDelegateTracerFallback:
    def test_no_active_tracer_uses_noop(self):
        provider = JudgmentTracerProvider.get_instance()
        delegate = provider._get_delegate_tracer()
        assert isinstance(delegate, NoOpTracer)


class TestAttachDetach:
    def test_attach_detach_round_trip(self, tracer):
        from opentelemetry.context import create_key, set_value, get_value

        provider = JudgmentTracerProvider.get_instance()
        key = create_key("test-key")
        ctx = set_value(key, "val")
        token = provider.attach_context(ctx)
        assert get_value(key, provider.get_current_context()) == "val"
        provider.detach_context(token)
