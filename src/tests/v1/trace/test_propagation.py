"""Tests for the propagation module — inject/extract roundtrip."""

from __future__ import annotations

from opentelemetry.context import Context

import judgeval.v1.trace.baggage as baggage
from judgeval.v1.trace import propagation


class TestPropagation:
    def test_inject_extract_roundtrip(self):
        ctx = Context()
        ctx = baggage.set_baggage("trace-key", "trace-value", ctx)

        carrier: dict[str, str] = {}
        propagation.inject(carrier, context=ctx)

        extracted = propagation.extract(carrier, context=Context())
        assert baggage.get_baggage("trace-key", extracted) == "trace-value"

    def test_get_set_global_textmap(self):
        original = propagation.get_global_textmap()
        assert original is not None

        from unittest.mock import MagicMock

        mock = MagicMock()
        propagation.set_global_textmap(mock)
        assert propagation.get_global_textmap() is mock

        # Restore
        propagation.set_global_textmap(original)
