"""Tests for IsolatedRandomIdGenerator to ensure immunity to global random.seed() calls."""

from __future__ import annotations

import random

from opentelemetry import trace
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from judgeval.v1.tracer.id_generator import IsolatedRandomIdGenerator


def test_generates_valid_span_id():
    generator = IsolatedRandomIdGenerator()
    span_id = generator.generate_span_id()

    assert span_id != trace.INVALID_SPAN_ID
    assert 0 < span_id < 2**64


def test_generates_valid_trace_id():
    generator = IsolatedRandomIdGenerator()
    trace_id = generator.generate_trace_id()

    assert trace_id != trace.INVALID_TRACE_ID
    assert 0 < trace_id < 2**128


def test_immune_to_global_random_seed():
    """Verify that random.seed() does not affect ID generation."""
    generator = IsolatedRandomIdGenerator()

    random.seed(42)
    trace_id_1 = generator.generate_trace_id()
    span_id_1 = generator.generate_span_id()

    random.seed(42)
    trace_id_2 = generator.generate_trace_id()
    span_id_2 = generator.generate_span_id()

    assert trace_id_1 != trace_id_2
    assert span_id_1 != span_id_2


def test_default_generator_affected_by_seed():
    """Verify that the default RandomIdGenerator IS affected by random.seed()."""
    generator = RandomIdGenerator()

    random.seed(42)
    trace_id_1 = generator.generate_trace_id()
    span_id_1 = generator.generate_span_id()

    random.seed(42)
    trace_id_2 = generator.generate_trace_id()
    span_id_2 = generator.generate_span_id()

    assert trace_id_1 == trace_id_2
    assert span_id_1 == span_id_2


def test_different_instances_produce_different_ids():
    """Verify that separate instances don't produce identical sequences."""
    gen1 = IsolatedRandomIdGenerator()
    gen2 = IsolatedRandomIdGenerator()

    ids1 = [gen1.generate_trace_id() for _ in range(10)]
    ids2 = [gen2.generate_trace_id() for _ in range(10)]

    assert ids1 != ids2


def test_generates_unique_ids():
    """Verify that generated IDs are unique over many iterations."""
    generator = IsolatedRandomIdGenerator()

    trace_ids = {generator.generate_trace_id() for _ in range(1000)}
    span_ids = {generator.generate_span_id() for _ in range(1000)}

    assert len(trace_ids) == 1000
    assert len(span_ids) == 1000
