"""
Microbenchmarks for SDK overhead.

Run with:
    pytest src/tests/reliability/benchmarks/ --benchmark-only
"""

from judgeval.v1.tracer.tracer import Tracer


class TestMicrobenchmarks:
    """Microbenchmarks for performance-critical operations."""

    def test_observe_decorator_overhead(self, benchmark, tracer: Tracer):
        """Benchmark @observe decorator overhead."""

        @tracer.observe(span_type="function")
        def traced_function():
            return "result"

        result = benchmark(traced_function)
        assert result == "result"

    def test_span_context_manager_overhead(self, benchmark, tracer: Tracer):
        """Benchmark span() context manager overhead."""

        def create_span():
            with tracer.span("test"):
                pass

        benchmark(create_span)

    def test_set_attribute_overhead(self, benchmark, tracer: Tracer):
        """Benchmark set_attribute overhead."""

        def set_attr():
            with tracer.span("test"):
                tracer.set_attribute("key", "value")

        benchmark(set_attr)
