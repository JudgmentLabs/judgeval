"""Tests for IsolatedRandomIdGenerator."""

from __future__ import annotations

import random

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from judgeval.v1.trace.id_generator import IsolatedRandomIdGenerator


def test_never_generates_invalid_ids():
    gen = IsolatedRandomIdGenerator()
    for _ in range(500):
        assert gen.generate_span_id() != trace_api.INVALID_SPAN_ID
        assert gen.generate_trace_id() != trace_api.INVALID_TRACE_ID


def test_generates_unique_ids():
    gen = IsolatedRandomIdGenerator()
    span_ids = {gen.generate_span_id() for _ in range(500)}
    trace_ids = {gen.generate_trace_id() for _ in range(500)}
    assert len(span_ids) == 500
    assert len(trace_ids) == 500


def test_immune_to_global_random_seed():
    gen = IsolatedRandomIdGenerator()
    std = RandomIdGenerator()

    random.seed(42)
    isolated_ids = [gen.generate_trace_id() for _ in range(5)]

    random.seed(42)
    isolated_ids2 = [gen.generate_trace_id() for _ in range(5)]

    # Isolated generator should NOT reproduce the same sequence after reseed
    assert isolated_ids != isolated_ids2

    # Standard generator DOES reproduce the same sequence
    random.seed(42)
    std_ids = [std.generate_trace_id() for _ in range(5)]
    random.seed(42)
    std_ids2 = [std.generate_trace_id() for _ in range(5)]
    assert std_ids == std_ids2
