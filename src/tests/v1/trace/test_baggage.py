"""Tests for baggage operations and JudgmentBaggagePropagator."""

from __future__ import annotations

from opentelemetry.context import Context

import judgeval.v1.trace.baggage as baggage
from judgeval.v1.trace.baggage.propagator import JudgmentBaggagePropagator


class TestBaggageOperations:
    def test_set_and_get(self):
        ctx = Context()
        ctx = baggage.set_baggage("key1", "value1", ctx)
        assert baggage.get_baggage("key1", ctx) == "value1"

    def test_get_missing_returns_none(self):
        ctx = Context()
        assert baggage.get_baggage("missing", ctx) is None

    def test_get_all(self):
        ctx = Context()
        ctx = baggage.set_baggage("a", "1", ctx)
        ctx = baggage.set_baggage("b", "2", ctx)
        all_baggage = baggage.get_all(ctx)
        assert dict(all_baggage) == {"a": "1", "b": "2"}

    def test_remove(self):
        ctx = Context()
        ctx = baggage.set_baggage("key", "val", ctx)
        ctx = baggage.remove_baggage("key", ctx)
        assert baggage.get_baggage("key", ctx) is None

    def test_clear(self):
        ctx = Context()
        ctx = baggage.set_baggage("a", "1", ctx)
        ctx = baggage.set_baggage("b", "2", ctx)
        ctx = baggage.clear(ctx)
        assert dict(baggage.get_all(ctx)) == {}

    def test_set_overwrites(self):
        ctx = Context()
        ctx = baggage.set_baggage("key", "old", ctx)
        ctx = baggage.set_baggage("key", "new", ctx)
        assert baggage.get_baggage("key", ctx) == "new"

    def test_remove_nonexistent_is_noop(self):
        ctx = Context()
        ctx = baggage.remove_baggage("nonexistent", ctx)
        assert dict(baggage.get_all(ctx)) == {}


class TestBaggageValidation:
    def test_valid_key(self):
        assert baggage._is_valid_key("simple-key") is True
        assert baggage._is_valid_key("key_with_underscore") is True

    def test_valid_value(self):
        assert baggage._is_valid_value("simple-value") is True

    def test_valid_pair(self):
        assert baggage._is_valid_pair("key", "value") is True


class TestJudgmentBaggagePropagator:
    def test_inject_and_extract_roundtrip(self):
        prop = JudgmentBaggagePropagator()
        ctx = Context()
        ctx = baggage.set_baggage("user-id", "abc-123", ctx)
        ctx = baggage.set_baggage("session", "xyz", ctx)

        carrier: dict[str, str] = {}
        prop.inject(carrier, ctx)
        assert "baggage" in carrier

        extracted_ctx = prop.extract(carrier, Context())
        assert baggage.get_baggage("user-id", extracted_ctx) == "abc-123"
        assert baggage.get_baggage("session", extracted_ctx) == "xyz"

    def test_extract_empty_carrier(self):
        prop = JudgmentBaggagePropagator()
        ctx = Context()
        result = prop.extract({}, ctx)
        assert dict(baggage.get_all(result)) == {}

    def test_inject_empty_baggage_does_not_set_header(self):
        prop = JudgmentBaggagePropagator()
        carrier: dict[str, str] = {}
        prop.inject(carrier, Context())
        assert "baggage" not in carrier

    def test_extract_oversized_header_ignored(self):
        prop = JudgmentBaggagePropagator()
        carrier = {"baggage": "k=" + "v" * 9000}
        ctx = prop.extract(carrier, Context())
        assert dict(baggage.get_all(ctx)) == {}

    def test_extract_oversized_pair_skipped(self):
        prop = JudgmentBaggagePropagator()
        long_pair = "longkey=" + "v" * 5000
        short_pair = "ok=fine"
        carrier = {"baggage": f"{long_pair},{short_pair}"}
        ctx = prop.extract(carrier, Context())
        assert baggage.get_baggage("ok", ctx) == "fine"
        assert baggage.get_baggage("longkey", ctx) is None

    def test_extract_malformed_entry_skipped(self):
        prop = JudgmentBaggagePropagator()
        carrier = {"baggage": "good=val,badentry,also=ok"}
        ctx = prop.extract(carrier, Context())
        assert baggage.get_baggage("good", ctx) == "val"
        assert baggage.get_baggage("also", ctx) == "ok"

    def test_fields_property(self):
        prop = JudgmentBaggagePropagator()
        assert prop.fields == {"baggage"}

    def test_url_encoding_roundtrip(self):
        prop = JudgmentBaggagePropagator()
        ctx = Context()
        ctx = baggage.set_baggage("key", "hello world", ctx)

        carrier: dict[str, str] = {}
        prop.inject(carrier, ctx)
        # Should be URL-encoded in the header
        assert "hello+world" in carrier["baggage"]

        extracted = prop.extract(carrier, Context())
        assert baggage.get_baggage("key", extracted) == "hello world"
