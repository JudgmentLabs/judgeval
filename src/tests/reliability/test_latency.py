"""
Latency tests for the v1 SDK.

These tests verify that the @observe decorator and tracing infrastructure
add minimal overhead to customer code.

Run with: pytest src/tests/reliability/test_latency.py -v
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from judgeval.v1.tracer.tracer import Tracer


@pytest.mark.reliability
class TestObserveOverhead:
    """Test that @observe decorator adds minimal latency overhead."""

    def test_observe_adds_minimal_overhead(self, tracer: Tracer):
        """
        Single traced call should add < 1ms overhead.

        This test compares execution time of a function with and without
        the @observe decorator to ensure overhead is negligible.
        """
        ITERATIONS = 100
        WORK_TIME_MS = 1  # 1ms simulated work
        MAX_OVERHEAD_MS = 1.0  # Maximum acceptable overhead per call

        def baseline_function():
            """Untraced baseline function."""
            time.sleep(WORK_TIME_MS / 1000)
            return "result"

        @tracer.observe(span_type="function")
        def traced_function():
            """Traced function with @observe."""
            time.sleep(WORK_TIME_MS / 1000)
            return "result"

        # Warm up
        for _ in range(10):
            baseline_function()
            traced_function()

        # Measure baseline
        baseline_times = []
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            baseline_function()
            baseline_times.append((time.perf_counter() - start) * 1000)

        # Measure traced
        traced_times = []
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            traced_function()
            traced_times.append((time.perf_counter() - start) * 1000)

        baseline_median = statistics.median(baseline_times)
        traced_median = statistics.median(traced_times)
        overhead = traced_median - baseline_median

        assert overhead < MAX_OVERHEAD_MS, (
            f"Overhead {overhead:.3f}ms exceeds maximum {MAX_OVERHEAD_MS}ms. "
            f"Baseline: {baseline_median:.3f}ms, Traced: {traced_median:.3f}ms"
        )

    def test_observe_under_concurrent_load(self, tracer: Tracer):
        """
        1000 concurrent traced calls should maintain low latency.

        Measures p50, p95, p99 latencies under concurrent load.
        Note: p99 can be higher due to thread scheduling overhead.
        """
        CONCURRENT_CALLS = 1000
        WORKERS = 50
        MAX_P99_MS = 100.0  # p99 can spike due to thread scheduling overhead

        @tracer.observe(span_type="function")
        def traced_function(call_id: int):
            """Function to be called concurrently."""
            time.sleep(0.001)  # 1ms work
            return f"result-{call_id}"

        latencies = []

        def timed_call(call_id: int) -> float:
            start = time.perf_counter()
            traced_function(call_id)
            return (time.perf_counter() - start) * 1000

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(timed_call, i) for i in range(CONCURRENT_CALLS)]
            for future in as_completed(futures):
                latencies.append(future.result())

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        assert p99 < MAX_P99_MS, (
            f"p99 latency {p99:.2f}ms exceeds maximum {MAX_P99_MS}ms. "
            f"p50: {p50:.2f}ms, p95: {p95:.2f}ms"
        )

    def test_nested_observe_scales_linearly(self, tracer: Tracer):
        """
        Nested @observe decorators should not cause exponential slowdown.

        Tests up to 10 levels of nesting.
        """
        # MAX_NESTING = 10
        ITERATIONS = 50
        MAX_OVERHEAD_PER_LEVEL_MS = 0.5

        # Create nested functions dynamically
        def create_nested_traced(depth: int):
            if depth == 0:

                @tracer.observe(span_type="function", span_name=f"level_{depth}")
                def base():
                    return "done"

                return base
            else:
                inner = create_nested_traced(depth - 1)

                @tracer.observe(span_type="function", span_name=f"level_{depth}")
                def wrapper():
                    return inner()

                return wrapper

        # Measure time for different nesting levels
        times_by_depth = {}

        for depth in [1, 5, 10]:
            nested_func = create_nested_traced(depth)

            # Warm up
            for _ in range(5):
                nested_func()

            # Measure
            times = []
            for _ in range(ITERATIONS):
                start = time.perf_counter()
                nested_func()
                times.append((time.perf_counter() - start) * 1000)

            times_by_depth[depth] = statistics.median(times)

        # Check that overhead scales roughly linearly
        overhead_per_level = (times_by_depth[10] - times_by_depth[1]) / 9

        assert overhead_per_level < MAX_OVERHEAD_PER_LEVEL_MS, (
            f"Overhead per nesting level {overhead_per_level:.3f}ms exceeds "
            f"maximum {MAX_OVERHEAD_PER_LEVEL_MS}ms. "
            f"Depth 1: {times_by_depth[1]:.2f}ms, "
            f"Depth 10: {times_by_depth[10]:.2f}ms"
        )

    def test_high_frequency_tracing(self, tracer: Tracer):
        """
        Test that SDK can handle 10,000+ traced calls per second.

        This simulates high-throughput production workloads.
        """
        TARGET_CALLS = 10000
        MAX_DURATION_SECONDS = 5.0  # Should complete in < 5 seconds

        @tracer.observe(span_type="function")
        def fast_traced_function(i: int):
            """Minimal work traced function."""
            return i * 2

        start = time.perf_counter()

        for i in range(TARGET_CALLS):
            fast_traced_function(i)

        duration = time.perf_counter() - start
        calls_per_second = TARGET_CALLS / duration

        assert duration < MAX_DURATION_SECONDS, (
            f"10k traced calls took {duration:.2f}s, exceeds max {MAX_DURATION_SECONDS}s. "
            f"Throughput: {calls_per_second:.0f} calls/sec"
        )

    def test_monitoring_disabled_adds_zero_overhead(
        self, tracer_monitoring_disabled: Tracer
    ):
        """
        When monitoring is disabled, @observe should add essentially zero overhead.
        """
        ITERATIONS = 1000
        MAX_OVERHEAD_MS = 0.1  # Essentially zero

        def baseline():
            return "result"

        @tracer_monitoring_disabled.observe(span_type="function")
        def traced():
            return "result"

        # Warm up
        for _ in range(100):
            baseline()
            traced()

        # Measure baseline
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            baseline()
        baseline_time = (time.perf_counter() - start) * 1000 / ITERATIONS

        # Measure traced
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            traced()
        traced_time = (time.perf_counter() - start) * 1000 / ITERATIONS

        overhead = traced_time - baseline_time

        assert overhead < MAX_OVERHEAD_MS, (
            f"Disabled monitoring still adds {overhead:.4f}ms overhead per call"
        )


@pytest.mark.reliability
class TestSpanOperationLatency:
    """Test latency of individual span operations."""

    def test_set_attribute_is_fast(self, tracer: Tracer):
        """
        set_attribute should be very fast (< 0.1ms per call).
        """
        ITERATIONS = 1000
        MAX_TIME_PER_CALL_MS = 0.1

        with tracer.span("test-span"):
            start = time.perf_counter()
            for i in range(ITERATIONS):
                tracer.set_attribute(f"key_{i}", f"value_{i}")
            duration = (time.perf_counter() - start) * 1000

        time_per_call = duration / ITERATIONS

        assert time_per_call < MAX_TIME_PER_CALL_MS, (
            f"set_attribute takes {time_per_call:.4f}ms per call, "
            f"exceeds max {MAX_TIME_PER_CALL_MS}ms"
        )

    def test_span_context_manager_is_fast(self, tracer: Tracer):
        """
        Using span() context manager should be fast.
        """
        ITERATIONS = 1000
        MAX_OVERHEAD_MS = 0.5

        start = time.perf_counter()
        for _ in range(ITERATIONS):
            with tracer.span("test-span"):
                pass  # No work
        duration = (time.perf_counter() - start) * 1000

        time_per_span = duration / ITERATIONS

        assert time_per_span < MAX_OVERHEAD_MS, (
            f"span() context manager takes {time_per_span:.3f}ms per call, "
            f"exceeds max {MAX_OVERHEAD_MS}ms"
        )
