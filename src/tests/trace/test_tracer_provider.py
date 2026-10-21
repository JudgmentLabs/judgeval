from __future__ import annotations

from unittest.mock import patch

from opentelemetry.trace import NoOpTracer

from judgeval.trace.tracer import Tracer
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider


def _make_tracer(**kwargs):
    defaults = dict(
        project_name="p", api_key="k", organization_id="o", api_url="http://x"
    )
    defaults.update(kwargs)
    with patch("judgeval.trace.tracer.resolve_project_id", return_value="pid"):
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
        assert t not in provider._judgment_tracers

    def test_set_active_returns_true(self):
        t = _make_tracer()
        provider = JudgmentTracerProvider.get_instance()
        result = provider.set_active(t)
        assert result is True

    def test_set_active_blocked_during_root_span(self, tracer, collecting_exporter):
        from judgeval.trace.base_tracer import BaseTracer

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
        from judgeval.trace.base_tracer import BaseTracer

        provider = JudgmentTracerProvider.get_instance()
        with BaseTracer.start_as_current_span("s"):
            span = provider.get_current_span()
            assert span.is_recording()


class TestHasActiveRootSpan:
    def test_false_when_no_span(self, tracer):
        assert JudgmentTracerProvider.get_instance().has_active_root_span() is False

    def test_true_at_root(self, tracer):
        from judgeval.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("root"):
            assert JudgmentTracerProvider.get_instance().has_active_root_span() is True

    def test_false_for_child_span(self, tracer):
        from judgeval.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("root"):
            with BaseTracer.start_as_current_span("child"):
                assert (
                    JudgmentTracerProvider.get_instance().has_active_root_span()
                    is False
                )


class TestGetTracer:
    def test_returns_proxy_tracer(self):
        from judgeval.trace.judgment_tracer_provider import ProxyTracer

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


class TestGlobalContextBridge:
    """When installed as the global provider, Judgment's active span is mirrored
    into the global OTel context (WRITE direction) so third-party
    instrumentation that reads the current span via trace.get_current_span()
    nests UNDER Judgment. Crucially, this is decoupled from the READ direction:
    Judgment still selects the parent for its own spans from its private
    context, so it never adopts a foreign ambient parent by default. When
    embedded (not the global provider), Judgment's context stays isolated."""

    def test_active_span_mirrored_to_global_for_external_nesting(
        self, tracer, collecting_exporter
    ):
        from opentelemetry import trace as otel_trace
        from judgeval.trace.base_tracer import BaseTracer

        provider = JudgmentTracerProvider.get_instance()
        # State after install_as_global_tracer_provider(): WRITE on, READ off.
        provider._mirror_active_span_to_global = True
        assert provider._use_global_context is False

        with BaseTracer.start_as_current_span("work"):
            external = otel_trace.get_current_span()  # reads the GLOBAL context
            assert external.is_recording()
            external.set_attribute("external.attr", "value")

        assert any(
            (s.attributes or {}).get("external.attr") == "value"
            for s in collecting_exporter.spans
        )

    def test_active_span_isolated_from_global_when_embedded(self, tracer):
        from opentelemetry import trace as otel_trace
        from judgeval.trace.base_tracer import BaseTracer

        provider = JudgmentTracerProvider.get_instance()
        assert provider._use_global_context is False  # default
        assert provider._mirror_active_span_to_global is False  # default

        with BaseTracer.start_as_current_span("work"):
            # Judgment's active span must not leak into the global context
            assert otel_trace.get_current_span().is_recording() is False

    def test_install_as_global_enables_write_not_read(self):
        from unittest.mock import patch

        provider = JudgmentTracerProvider.get_instance()
        assert provider._use_global_context is False
        assert provider._mirror_active_span_to_global is False
        with (
            patch.object(JudgmentTracerProvider, "get_instance", return_value=provider),
            patch(
                "judgeval.trace.judgment_tracer_provider.trace_api.set_tracer_provider"
            ),
            patch(
                "judgeval.trace.judgment_tracer_provider.trace_api.get_tracer_provider",
                return_value=provider,
            ),
        ):
            assert JudgmentTracerProvider.install_as_global_tracer_provider() is True
        # WRITE enabled (external libs nest under Judgment)...
        assert provider._mirror_active_span_to_global is True
        # ...but READ (foreign-parent adoption) stays off by default.
        assert provider._use_global_context is False


class TestForeignParentDecoupling:
    """Regression tests for the #749 fix: Judgment must not adopt a foreign
    ambient parent sitting in the global OTel context (the Google ADK / Vertex
    Agent Engine scenario), while still exporting its own spans and preserving
    intra-Judgment nesting."""

    @staticmethod
    def _foreign_unsampled_global_context():
        """Attach an unsampled foreign span into the GLOBAL OTel context,
        mimicking an ambient ADK/Vertex request span. Returns
        (foreign_span_context, detach_token)."""
        from opentelemetry import context as context_api
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import (
            NonRecordingSpan,
            SpanContext,
            TraceFlags,
        )

        foreign_ctx = SpanContext(
            trace_id=0x11111111111111111111111111111111,
            span_id=0x2222222222222222,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.DEFAULT),  # NOT sampled
        )
        assert foreign_ctx.trace_flags.sampled is False
        foreign_span = NonRecordingSpan(foreign_ctx)
        global_ctx = otel_trace.set_span_in_context(foreign_span)
        token = context_api.attach(global_ctx)
        return foreign_ctx, token

    def test_span_under_unsampled_ambient_parent_still_exports_as_judgment_root(
        self, tracer, collecting_exporter
    ):
        from opentelemetry import context as context_api
        from judgeval.trace.base_tracer import BaseTracer

        provider = JudgmentTracerProvider.get_instance()
        # Simulate the production scenario: Judgment is installed as the global
        # provider (WRITE mirroring on) and a foreign, unsampled parent is
        # already active in the global context.
        provider._mirror_active_span_to_global = True

        foreign_ctx, token = self._foreign_unsampled_global_context()
        try:
            with BaseTracer.start_as_current_span("judgment-root") as span:
                sc = span.get_span_context()
                # (a) recording / sampled despite the unsampled ambient parent
                assert span.is_recording()
                assert sc.trace_flags.sampled is True
                # (c) independent Judgment ROOT, not re-parented under the foreign span
                assert sc.trace_id != foreign_ctx.trace_id
                assert getattr(span, "parent", None) is None
        finally:
            context_api.detach(token)

        # (b) the Judgment span was exported
        assert len(collecting_exporter.spans) >= 1
        exported = collecting_exporter.spans[-1]
        assert exported.parent is None
        assert exported.context.trace_id != foreign_ctx.trace_id

    def test_intra_judgment_nesting_preserved(self, tracer):
        from judgeval.trace.base_tracer import BaseTracer

        provider = JudgmentTracerProvider.get_instance()
        provider._mirror_active_span_to_global = True

        with BaseTracer.start_as_current_span("parent") as parent:
            parent_sc = parent.get_span_context()
            with BaseTracer.start_as_current_span("child") as child:
                child_sc = child.get_span_context()
                # child correctly nests under the Judgment parent
                assert child_sc.trace_id == parent_sc.trace_id
                assert child.parent is not None
                assert child.parent.span_id == parent_sc.span_id

    def test_opt_in_use_global_context_restores_adoption(
        self, tracer, collecting_exporter
    ):
        """With the explicit opt-in, Judgment reads the global context and DOES
        adopt the ambient parent (restores full #749 read behavior)."""
        from opentelemetry import context as context_api
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan
        from judgeval.trace.base_tracer import BaseTracer

        provider = JudgmentTracerProvider.get_instance()
        provider._use_global_context = True  # explicit opt-in

        # Use a SAMPLED foreign parent here so the child is recorded.
        foreign_ctx = SpanContext(
            trace_id=0x33333333333333333333333333333333,
            span_id=0x4444444444444444,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        token = context_api.attach(
            otel_trace.set_span_in_context(NonRecordingSpan(foreign_ctx))
        )
        try:
            with BaseTracer.start_as_current_span("adopted") as span:
                sc = span.get_span_context()
                # adopts the foreign trace and parents under the foreign span
                assert sc.trace_id == foreign_ctx.trace_id
                assert span.parent is not None
                assert span.parent.span_id == foreign_ctx.span_id
        finally:
            context_api.detach(token)
