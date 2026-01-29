"""
Sustained load tests for the v1 SDK.

These tests verify stability under sustained high load.

Run with: pytest src/tests/reliability/test_sustained_load.py -v
"""

import gc
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest

from judgeval.v1.tracer.tracer import Tracer


@pytest.mark.reliability
@pytest.mark.slow
class TestSustainedLoad:
    """Test SDK behavior under sustained high load."""

    def test_100k_spans_over_60_seconds(self, tracer: Tracer):
        """
        Generate 100k spans over 60 seconds (~1600 spans/sec).
        """
        TARGET_SPANS = 100_000
        DURATION_SECONDS = 60
        MAX_MEMORY_GROWTH_MB = 50

        @tracer.observe(span_type="function")
        def traced_function(i: int):
            return f"result-{i}"

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(stat.size for stat in baseline.statistics("filename"))

        start_time = time.perf_counter()
        spans_per_second = TARGET_SPANS / DURATION_SECONDS

        for i in range(TARGET_SPANS):
            traced_function(i)
            elapsed = time.perf_counter() - start_time
            expected_elapsed = (i + 1) / spans_per_second
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)

        tracer.force_flush(timeout_millis=30000)

        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(stat.size for stat in final.statistics("filename"))
        tracemalloc.stop()

        memory_growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert memory_growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew {memory_growth_mb:.1f}MB over 60s "
            f"(max: {MAX_MEMORY_GROWTH_MB}MB)"
        )

    def test_burst_traffic_spike(self, tracer: Tracer):
        """
        Simulate sudden traffic spike: 10k spans in 1 second.
        """
        BURST_SIZE = 10_000
        MAX_MEMORY_GROWTH_MB = 30

        @tracer.observe(span_type="function")
        def fast_function(i: int):
            return i * 2

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(stat.size for stat in baseline.statistics("filename"))

        # start = time.perf_counter()
        for i in range(BURST_SIZE):
            fast_function(i)
        # burst_duration = time.perf_counter() - start

        time.sleep(2)
        tracer.force_flush(timeout_millis=30000)

        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(stat.size for stat in final.statistics("filename"))
        tracemalloc.stop()

        memory_growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert memory_growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew {memory_growth_mb:.1f}MB after burst "
            f"(max: {MAX_MEMORY_GROWTH_MB}MB)"
        )

    def test_queue_saturation_under_load(self, tracer: Tracer):
        """
        Generate spans faster than export capacity.
        """

        @tracer.observe(span_type="function")
        def rapid_function(i: int):
            return i

        for i in range(5000):
            rapid_function(i)

        assert True  # If we get here, no crash

    def test_concurrent_high_frequency_threads(self, tracer: Tracer):
        """
        100 threads each generating 1000 spans concurrently.
        """
        THREADS = 100
        SPANS_PER_THREAD = 1000

        @tracer.observe(span_type="function")
        def thread_function(thread_id: int, iteration: int):
            return f"{thread_id}-{iteration}"

        def worker(thread_id: int):
            for i in range(SPANS_PER_THREAD):
                thread_function(thread_id, i)

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = [executor.submit(worker, i) for i in range(THREADS)]
            for future in futures:
                future.result()

        tracer.force_flush(timeout_millis=60000)

        assert True

    @pytest.mark.skip(reason="Long-running test, run manually")
    def test_long_running_service_24_hours(self, tracer: Tracer):
        """
        Run tracer for 24 hours with steady load.
        """
        import psutil
        import os

        DURATION_HOURS = 24
        SPANS_PER_MINUTE = 100

        @tracer.observe(span_type="function")
        def steady_function(i: int):
            return i

        process = psutil.Process(os.getpid())
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024

        start_time = time.time()
        iteration = 0

        while time.time() - start_time < DURATION_HOURS * 3600:
            for _ in range(SPANS_PER_MINUTE):
                steady_function(iteration)
                iteration += 1

            time.sleep(60)

            if iteration % (SPANS_PER_MINUTE * 60) == 0:
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory_mb - baseline_memory_mb
                print(
                    f"Hour {iteration // (SPANS_PER_MINUTE * 60)}: "
                    f"Memory growth: {memory_growth:.1f}MB"
                )

                assert memory_growth < 100, "Memory leak detected"
