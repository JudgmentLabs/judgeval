from __future__ import annotations


import judgeval.v1.trace.baggage as baggage
from judgeval.v1.trace.baggage.propagator import JudgmentBaggagePropagator


def _fresh_ctx():
    from judgeval.v1.trace.judgment_tracer_provider import JudgmentTracerProvider

    return JudgmentTracerProvider.get_instance().get_current_context()


class TestBaggageOperations:
    def test_set_and_get(self):
        ctx = baggage.set_baggage("k", "v", _fresh_ctx())
        assert baggage.get_baggage("k", ctx) == "v"

    def test_get_all(self):
        ctx = _fresh_ctx()
        ctx = baggage.set_baggage("a", "1", ctx)
        ctx = baggage.set_baggage("b", "2", ctx)
        result = baggage.get_all(ctx)
        assert result["a"] == "1"
        assert result["b"] == "2"

    def test_remove_baggage(self):
        ctx = baggage.set_baggage("key", "val", _fresh_ctx())
        ctx = baggage.remove_baggage("key", ctx)
        assert baggage.get_baggage("key", ctx) is None

    def test_clear(self):
        ctx = baggage.set_baggage("a", "1", _fresh_ctx())
        ctx = baggage.set_baggage("b", "2", ctx)
        ctx = baggage.clear(ctx)
        assert dict(baggage.get_all(ctx)) == {}

    def test_get_missing_key_returns_none(self):
        assert baggage.get_baggage("missing", _fresh_ctx()) is None

    def test_overwrite_value(self):
        ctx = baggage.set_baggage("k", "old", _fresh_ctx())
        ctx = baggage.set_baggage("k", "new", ctx)
        assert baggage.get_baggage("k", ctx) == "new"


class TestBaggageValidation:
    def test_valid_key_and_value(self):
        assert baggage._is_valid_pair("valid-key", "valid-value")

    def test_invalid_key_with_spaces(self):
        assert not baggage._is_valid_pair("invalid key", "value")

    def test_empty_key_invalid(self):
        assert not baggage._is_valid_pair("", "value")


class TestJudgmentBaggagePropagator:
    def test_fields_contains_baggage_header(self):
        prop = JudgmentBaggagePropagator()
        assert "baggage" in prop.fields

    def test_inject_sets_baggage_header(self):
        prop = JudgmentBaggagePropagator()
        ctx = baggage.set_baggage("key", "val", _fresh_ctx())
        carrier = {}
        prop.inject(carrier, context=ctx)
        assert "baggage" in carrier
        assert "key" in carrier["baggage"]

    def test_extract_reads_baggage_header(self):
        prop = JudgmentBaggagePropagator()
        carrier = {"baggage": "mykey=myval"}
        ctx = prop.extract(carrier, context=_fresh_ctx())
        assert baggage.get_baggage("mykey", ctx) == "myval"

    def test_inject_no_baggage_skips_header(self):
        prop = JudgmentBaggagePropagator()
        carrier = {}
        prop.inject(carrier, context=_fresh_ctx())
        assert "baggage" not in carrier

    def test_extract_empty_carrier_returns_context(self):
        prop = JudgmentBaggagePropagator()
        ctx = prop.extract({}, context=_fresh_ctx())
        assert ctx is not None

    def test_extract_oversized_header_returns_unchanged_context(self):
        prop = JudgmentBaggagePropagator()
        big = "k=v," * 2049
        carrier = {"baggage": big}
        ctx = prop.extract(carrier, context=_fresh_ctx())
        assert ctx is not None
