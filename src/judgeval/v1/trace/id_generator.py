from __future__ import annotations

import random

from opentelemetry import trace
from opentelemetry.sdk.trace.id_generator import IdGenerator


class IsolatedRandomIdGenerator(IdGenerator):
    """Generates trace and span IDs using an isolated ``Random`` instance.

        Unlike the default OTel generator, this is immune to ``random.seed()``
    calls in application code. Prevents ID collisions in forking servers
        like FastAPI on Uvicorn where child processes may share RNG state.
    """

    __slots__ = ("_random",)

    def __init__(self):
        self._random = random.Random()

    def generate_span_id(self) -> int:
        """Generate a random 64-bit span ID."""
        span_id = self._random.getrandbits(64)
        while span_id == trace.INVALID_SPAN_ID:
            span_id = self._random.getrandbits(64)
        return span_id

    def generate_trace_id(self) -> int:
        """Generate a random 128-bit trace ID."""
        trace_id = self._random.getrandbits(128)
        while trace_id == trace.INVALID_TRACE_ID:
            trace_id = self._random.getrandbits(128)
        return trace_id
