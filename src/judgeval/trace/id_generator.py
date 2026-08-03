from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.trace.id_generator import IdGenerator


class IsolatedRandomIdGenerator(IdGenerator):
    """Generates trace and span IDs using ``os.urandom()`` for fork-safe entropy.

    ``os.urandom()`` reads directly from the OS entropy pool on every call and
    never shares state across forked processes, making this generator safe for
    use in forking servers such as Uvicorn (``--workers N``) or Gunicorn
    pre-fork workers where ``Tracer.init()`` is called before the fork.

    A ``random.Random`` instance seeded before a fork is **not** fork-safe:
    every child process inherits an identical copy of the PRNG state and
    therefore generates the same ID sequence, causing collisions.
    """

    def generate_span_id(self) -> int:
        """Generate a random 64-bit span ID."""
        while True:
            span_id = int.from_bytes(os.urandom(8), "big")
            if span_id != trace.INVALID_SPAN_ID:
                return span_id

    def generate_trace_id(self) -> int:
        """Generate a random 128-bit trace ID."""
        while True:
            trace_id = int.from_bytes(os.urandom(16), "big")
            if trace_id != trace.INVALID_TRACE_ID:
                return trace_id
