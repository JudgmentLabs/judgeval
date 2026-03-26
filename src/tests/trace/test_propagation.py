from __future__ import annotations


import judgeval.trace.propagation as propagation
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


class TestGlobalTextmap:
    def test_default_is_composite(self):
        from opentelemetry.propagators.composite import CompositePropagator

        assert isinstance(propagation.get_global_textmap(), CompositePropagator)

    def test_set_and_get_custom_propagator(self):
        original = propagation.get_global_textmap()
        custom = TraceContextTextMapPropagator()
        try:
            propagation.set_global_textmap(custom)
            assert propagation.get_global_textmap() is custom
        finally:
            propagation.set_global_textmap(original)


class TestInjectExtract:
    def test_inject_extract_round_trip(self, tracer):
        from judgeval.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("root"):
            carrier = {}
            propagation.inject(carrier)
            assert "traceparent" in carrier

            ctx = propagation.extract(carrier)
            assert ctx is not None

    def test_extract_empty_carrier_returns_context(self, tracer):
        ctx = propagation.extract({})
        assert ctx is not None

    def test_inject_with_explicit_context(self, tracer):
        from judgeval.trace.base_tracer import BaseTracer

        with BaseTracer.start_as_current_span("root"):
            from judgeval.trace.judgment_tracer_provider import (
                JudgmentTracerProvider,
            )

            ctx = JudgmentTracerProvider.get_instance().get_current_context()
            carrier = {}
            propagation.inject(carrier, context=ctx)
            assert "traceparent" in carrier


class TestBaggagePropagation:
    def test_inject_extract_baggage_round_trip(self, tracer):
        import judgeval.trace.baggage as baggage

        from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

        ctx = JudgmentTracerProvider.get_instance().get_current_context()
        ctx = baggage.set_baggage("user_id", "abc123", ctx)

        carrier = {}
        propagation.inject(carrier, context=ctx)

        extracted = propagation.extract(carrier)
        assert baggage.get_baggage("user_id", extracted) == "abc123"
