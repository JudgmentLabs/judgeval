from __future__ import annotations

import random

from opentelemetry.trace import INVALID_SPAN_ID, INVALID_TRACE_ID

from judgeval.v1.trace.id_generator import IsolatedRandomIdGenerator


def test_generates_unique_span_ids():
    gen = IsolatedRandomIdGenerator()
    ids = {gen.generate_span_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_generates_unique_trace_ids():
    gen = IsolatedRandomIdGenerator()
    ids = {gen.generate_trace_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_span_id_never_invalid():
    gen = IsolatedRandomIdGenerator()
    for _ in range(1000):
        assert gen.generate_span_id() != INVALID_SPAN_ID


def test_trace_id_never_invalid():
    gen = IsolatedRandomIdGenerator()
    for _ in range(1000):
        assert gen.generate_trace_id() != INVALID_TRACE_ID


def test_immune_to_global_seed():
    gen = IsolatedRandomIdGenerator()
    random.seed(42)
    ids_seeded = [gen.generate_span_id() for _ in range(10)]
    random.seed(42)
    ids_seeded_again = [gen.generate_span_id() for _ in range(10)]
    assert ids_seeded != ids_seeded_again


def test_different_instances_produce_different_sequences():
    g1 = IsolatedRandomIdGenerator()
    g2 = IsolatedRandomIdGenerator()
    ids1 = [g1.generate_trace_id() for _ in range(10)]
    ids2 = [g2.generate_trace_id() for _ in range(10)]
    assert ids1 != ids2
