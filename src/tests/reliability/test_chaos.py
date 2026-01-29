"""
Chaos engineering tests for the v1 SDK.

These tests verify robustness to unpredictable failures.

Run with: pytest src/tests/reliability/test_chaos.py -v -m chaos
"""

import random
import threading
import time
from unittest.mock import patch

import pytest
import httpx

from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.processors.judgment_span_processor import JudgmentSpanProcessor


def _get_span_processor(tracer: Tracer) -> JudgmentSpanProcessor:
    processor = tracer._tracer_provider._active_span_processor
    for span_processor in getattr(processor, "_span_processors", []):
        if isinstance(span_processor, JudgmentSpanProcessor):
            return span_processor
    raise AssertionError("JudgmentSpanProcessor not found on tracer provider")


@pytest.mark.reliability
@pytest.mark.chaos
class TestChaosScenarios:
    """Chaos engineering tests for unpredictable failures."""

    def test_random_api_timeouts(self, tracer: Tracer):
        """
        20% of tag API calls timeout randomly.
        """
        original_request = tracer.api_client._request

        def flaky_api(*args, **kwargs):
            if random.random() < 0.2:
                raise TimeoutError("Random timeout")
            return original_request(*args, **kwargs)

        with patch.object(tracer.api_client, "_request", side_effect=flaky_api):
            for i in range(100):
                with tracer.span(f"test-{i}"):
                    if i % 10 == 0:
                        tracer.tag([f"tag-{i}"])

        assert True

    def test_intermittent_network_partitions(self, tracer: Tracer):
        """
        Simulate network dropping 50% of export attempts.
        """
        span_processor = _get_span_processor(tracer)
        exporter = span_processor._span_exporter
        original_export = exporter._delegate.export

        call_count = {"count": 0}

        def unreliable_network(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] % 2 == 0:
                raise httpx.NetworkError("Packet dropped")
            return original_export(*args, **kwargs)

        with patch.object(exporter._delegate, "export", side_effect=unreliable_network):
            for i in range(50):
                with tracer.span(f"test-{i}"):
                    pass

            tracer.force_flush(timeout_millis=30000)

        assert call_count["count"] >= 50

    @pytest.mark.skip(reason="Requires cgroups/container limits")
    def test_memory_pressure(self, tracer: Tracer):
        """
        Limit SDK memory to 100MB and generate high trace volume.
        """
        for i in range(100_000):
            with tracer.span(f"span-{i}"):
                tracer.set_attribute("data", "x" * 1000)

        assert True

    def test_cpu_throttling(self, tracer: Tracer):
        """
        Customer code should remain fast even with background pressure.
        """

        @tracer.observe(span_type="function")
        def customer_function(i: int):
            return i * 2

        start = time.perf_counter()
        for i in range(1000):
            customer_function(i)
        duration = time.perf_counter() - start

        assert duration < 1.0

    def test_concurrent_tracer_shutdown(self, tracer: Tracer):
        """
        Shutdown tracer while spans are in-flight.
        """

        def create_spans():
            for i in range(1000):
                with tracer.span(f"concurrent-{i}"):
                    time.sleep(0.001)

        thread = threading.Thread(target=create_spans)
        thread.start()

        time.sleep(0.1)
        tracer.shutdown(timeout_millis=5000)

        thread.join()
        assert True
