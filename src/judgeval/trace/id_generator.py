from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.trace.id_generator import IdGenerator


class IsolatedRandomIdGenerator(IdGenerator):
    """Generates trace and span IDs from the OS entropy pool via ``os.urandom()``.

    Each call reads fresh bytes directly from the operating system, so IDs are
    statistically unique across threads, processes, and forked workers without
    any shared or inherited state.
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
