"""
Memory tests for the v1 SDK.

These tests verify that the SDK does not leak memory over extended use,
properly cleans up spans, and releases resources on shutdown.

Run with: pytest src/tests/reliability/test_memory.py -v
"""

import pytest
import gc
import tracemalloc
from typing import List
from unittest.mock import patch

from judgeval.v1.tracer.tracer import Tracer


@pytest.mark.reliability
class TestMemoryStability:
    """Test that SDK maintains stable memory usage over time."""

    def test_no_memory_leak_over_many_calls(self, tracer: Tracer):
        """
        100k traced calls should not cause unbounded memory growth.

        Memory growth should stay under 50MB after 100k calls.
        """
        ITERATIONS = 100_000
        MAX_MEMORY_GROWTH_MB = 50
        CHECKPOINT_INTERVAL = 10_000

        @tracer.observe(span_type="function")
        def traced_function(i: int):
            return f"result-{i}"

        # Warm up and establish baseline
        for i in range(1000):
            traced_function(i)

        gc.collect()
        tracemalloc.start()

        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_size = sum(
            stat.size for stat in baseline_snapshot.statistics("filename")
        )

        memory_checkpoints: List[float] = []

        for i in range(ITERATIONS):
            traced_function(i)

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                gc.collect()
                current_snapshot = tracemalloc.take_snapshot()
                current_size = sum(
                    stat.size for stat in current_snapshot.statistics("filename")
                )
                growth_mb = (current_size - baseline_size) / 1024 / 1024
                memory_checkpoints.append(growth_mb)

        tracemalloc.stop()
        gc.collect()

        final_growth = memory_checkpoints[-1] if memory_checkpoints else 0

        assert final_growth < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew by {final_growth:.1f}MB after {ITERATIONS} calls, "
            f"exceeds max {MAX_MEMORY_GROWTH_MB}MB. "
            f"Checkpoints (MB): {[f'{c:.1f}' for c in memory_checkpoints]}"
        )

    def test_span_cleanup_on_exception(self, tracer: Tracer):
        """
        Spans should be properly cleaned up even when user code throws exceptions.

        This tests that the SDK doesn't leak span objects on error paths.
        """
        ITERATIONS = 1000
        MAX_MEMORY_GROWTH_MB = 5

        @tracer.observe(span_type="function")
        def failing_function(i: int):
            if i % 2 == 0:
                raise ValueError(f"Error at {i}")
            return f"result-{i}"

        gc.collect()
        tracemalloc.start()
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_size = sum(
            stat.size for stat in baseline_snapshot.statistics("filename")
        )

        for i in range(ITERATIONS):
            try:
                failing_function(i)
            except ValueError:
                pass  # Expected

        gc.collect()
        final_snapshot = tracemalloc.take_snapshot()
        final_size = sum(stat.size for stat in final_snapshot.statistics("filename"))
        tracemalloc.stop()

        growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew by {growth_mb:.1f}MB after {ITERATIONS} exceptions, "
            f"suggesting span leak on error path"
        )

    def test_manual_span_cleanup(self, tracer: Tracer):
        """
        Manually created spans via span() context manager should be cleaned up.
        """
        ITERATIONS = 1000
        MAX_MEMORY_GROWTH_MB = 5

        gc.collect()
        tracemalloc.start()
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_size = sum(
            stat.size for stat in baseline_snapshot.statistics("filename")
        )

        for i in range(ITERATIONS):
            with tracer.span(f"test-span-{i}"):
                tracer.set_input({"index": i})
                tracer.set_output({"result": i * 2})

        gc.collect()
        final_snapshot = tracemalloc.take_snapshot()
        final_size = sum(stat.size for stat in final_snapshot.statistics("filename"))
        tracemalloc.stop()

        growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew by {growth_mb:.1f}MB after {ITERATIONS} manual spans"
        )

    def test_tracer_shutdown_releases_resources(self, mock_client):
        """
        Calling shutdown() should release tracer resources.
        """
        from judgeval.v1.tracer.tracer_factory import TracerFactory

        gc.collect()
        tracemalloc.start()

        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            factory = TracerFactory(mock_client)
            tracer = factory.create(
                project_name="shutdown-test",
                enable_monitoring=True,
                enable_evaluation=False,
                isolated=True,
            )

            @tracer.observe(span_type="function")
            def traced_function():
                return "result"

            # Create some spans
            for _ in range(100):
                traced_function()

            pre_shutdown = tracemalloc.take_snapshot()
            pre_size = sum(stat.size for stat in pre_shutdown.statistics("filename"))

            # Shutdown tracer
            tracer.force_flush(timeout_millis=5000)
            tracer.shutdown(timeout_millis=5000)

        gc.collect()
        post_shutdown = tracemalloc.take_snapshot()
        post_size = sum(stat.size for stat in post_shutdown.statistics("filename"))
        tracemalloc.stop()

        # Memory should not increase after shutdown (ideally should decrease)
        # We just check it doesn't grow significantly
        growth_mb = (post_size - pre_size) / 1024 / 1024

        assert growth_mb < 5, (
            f"Memory grew by {growth_mb:.1f}MB after shutdown, "
            "resources may not be released properly"
        )


@pytest.mark.reliability
class TestLargePayloadMemory:
    """Test memory behavior with large input/output payloads."""

    def test_large_payloads_dont_accumulate(self, tracer: Tracer):
        """
        Large payloads should be serialized and not held in memory.
        """
        ITERATIONS = 100
        PAYLOAD_SIZE_BYTES = 100_000  # 100KB per payload
        MAX_MEMORY_GROWTH_MB = 50

        @tracer.observe(span_type="function")
        def function_with_large_payload():
            large_input = "x" * PAYLOAD_SIZE_BYTES
            return large_input

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(stat.size for stat in baseline.statistics("filename"))

        for _ in range(ITERATIONS):
            function_with_large_payload()

        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(stat.size for stat in final.statistics("filename"))
        tracemalloc.stop()

        growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew by {growth_mb:.1f}MB with large payloads, "
            f"suggests payloads are being held in memory"
        )

    def test_attributes_with_large_values(self, tracer: Tracer):
        """
        Setting large attribute values should not cause memory issues.
        """
        ITERATIONS = 100
        ATTRIBUTE_SIZE_BYTES = 10_000  # 10KB per attribute
        MAX_MEMORY_GROWTH_MB = 20

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(stat.size for stat in baseline.statistics("filename"))

        for i in range(ITERATIONS):
            with tracer.span(f"span-{i}"):
                large_value = {"data": "x" * ATTRIBUTE_SIZE_BYTES}
                tracer.set_attribute("large_attr", large_value)

        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(stat.size for stat in final.statistics("filename"))
        tracemalloc.stop()

        growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew by {growth_mb:.1f}MB with large attributes"
        )


@pytest.mark.reliability
class TestMemoryUnderStress:
    """Test memory behavior under stress conditions."""

    def test_rapid_span_creation_and_destruction(self, tracer: Tracer):
        """
        Rapidly creating and destroying spans should not leak memory.
        """
        ITERATIONS = 10_000
        MAX_MEMORY_GROWTH_MB = 20

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(stat.size for stat in baseline.statistics("filename"))

        for i in range(ITERATIONS):
            with tracer.span(f"rapid-span-{i}"):
                pass  # Immediate exit

        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(stat.size for stat in final.statistics("filename"))
        tracemalloc.stop()

        growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew by {growth_mb:.1f}MB during rapid span creation"
        )

    def test_deeply_nested_spans_memory(self, tracer: Tracer):
        """
        Deeply nested spans should not cause excessive memory usage.
        """
        NESTING_DEPTH = 50
        ITERATIONS = 100
        MAX_MEMORY_GROWTH_MB = 20

        def create_nested_spans(depth: int):
            if depth == 0:
                return
            with tracer.span(f"nested-{depth}"):
                create_nested_spans(depth - 1)

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(stat.size for stat in baseline.statistics("filename"))

        for _ in range(ITERATIONS):
            create_nested_spans(NESTING_DEPTH)

        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(stat.size for stat in final.statistics("filename"))
        tracemalloc.stop()

        growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew by {growth_mb:.1f}MB with deeply nested spans"
        )
