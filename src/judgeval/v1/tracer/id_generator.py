from __future__ import annotations

import random

from opentelemetry import trace
from opentelemetry.sdk.trace.id_generator import IdGenerator


class IsolatedRandomIdGenerator(IdGenerator):
    """ID generator using an isolated random instance immune to global random.seed() calls."""

    __slots__ = ("_random",)

    def __init__(self):
        self._random = random.Random()

    def generate_span_id(self) -> int:
        span_id = self._random.getrandbits(64)
        while span_id == trace.INVALID_SPAN_ID:
            span_id = self._random.getrandbits(64)
        return span_id

    def generate_trace_id(self) -> int:
        trace_id = self._random.getrandbits(128)
        while trace_id == trace.INVALID_TRACE_ID:
            trace_id = self._random.getrandbits(128)
        return trace_id
