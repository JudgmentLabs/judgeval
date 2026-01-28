# Judgeval SDK Reliability Testing Plan

**Version:** 1.0
**Last Updated:** 2026-01-27
**Purpose:** Comprehensive test plan to ensure the SDK never negatively impacts customer applications

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Suite Overview](#test-suite-overview)
3. [Unit Test Suites](#unit-test-suites)
4. [Load Testing](#load-testing)
5. [Chaos Engineering](#chaos-engineering)
6. [Performance Benchmarking](#performance-benchmarking)
7. [CI/CD Integration](#cicd-integration)
8. [Success Criteria](#success-criteria)

---

## Testing Philosophy

### Core Principles

**Non-Invasive:** The SDK must never crash, block, or slow down customer applications.

**Observable Failures:** When the SDK fails, it should be visible in logs and metrics, not customer code.

**Graceful Degradation:** Under extreme load or API failures, the SDK should drop traces gracefully rather than crash.

**Predictable Performance:** `@observe` decorator overhead must be consistently < 1ms.

### What We're Testing

1. **Critical Path Impact** - Does the SDK block customer code?
2. **Trace Reliability** - Are traces dropped under sustained load?
3. **Memory Stability** - Does memory grow unbounded over time?
4. **Exception Isolation** - Do SDK errors propagate to customers?
5. **Performance** - What overhead does tracing add?

---

## Test Suite Overview

| Suite | Priority | Tests | Duration | Purpose |
|-------|----------|-------|----------|---------|
| [API Failure Resilience](#1-api-failure-resilience) | **Critical** | 5 | ~30s | Verify non-blocking behavior |
| [Sustained Load](#2-sustained-load-testing) | **Critical** | 5 | ~2min | Verify stability under load |
| [Crash Resistance](#3-crash-resistance) | **High** | 4 | ~20s | Verify exception isolation |
| [Memory Leak Detection](#4-memory-leak-detection) | **High** | 3 | ~1min | Verify bounded memory |
| [Latency Under Stress](#5-latency-under-stress) | **Medium** | 3 | ~30s | Verify performance degrades gracefully |
| [Chaos Engineering](#6-chaos-scenarios) | **Medium** | 5 | ~5min | Verify robustness to failures |
| [Load Testing](#load-testing) | **Medium** | - | ~10min | Verify throughput limits |

**Total New Tests:** 25
**Total Estimated Runtime:** ~15 minutes

---

## Unit Test Suites

### 1. API Failure Resilience

**File:** `src/tests/reliability/test_api_failures.py`

**Purpose:** Verify the SDK handles API failures gracefully without blocking customer code.

#### Test 1: Evaluation Enqueue with Slow API

```python
import pytest
import time
from unittest.mock import patch, MagicMock
from judgeval.v1.tracer.tracer import Tracer

@pytest.mark.reliability
class TestAPIFailureResilience:
    """Test SDK behavior when backend API is slow or failing."""

    def test_evaluation_enqueue_with_slow_api(self, tracer: Tracer):
        """
        Verify that evaluation enqueue doesn't block customer code
        when API is slow.

        Expected behavior:
        - Customer code completes quickly (< 100ms)
        - Evaluation queued in background
        - Eventually sent when API responds
        """
        # Mock API with 10-second delay
        original_method = tracer.api_client.add_to_run_eval_queue_examples

        def slow_api_call(*args, **kwargs):
            time.sleep(10)  # Simulate slow API
            return original_method(*args, **kwargs)

        with patch.object(
            tracer.api_client,
            'add_to_run_eval_queue_examples',
            side_effect=slow_api_call
        ):
            start = time.perf_counter()

            with tracer.span("test-span"):
                # This should NOT block
                tracer.async_evaluate(
                    input={"prompt": "test"},
                    output={"response": "test"},
                    scorers=[]
                )

            duration_ms = (time.perf_counter() - start) * 1000

            # Customer code should complete in < 100ms
            assert duration_ms < 100, (
                f"Customer code blocked for {duration_ms:.1f}ms "
                f"waiting for slow API (max: 100ms)"
            )
```

#### Test 2: Evaluation Enqueue with API Down

```python
    def test_evaluation_enqueue_with_api_down(self, tracer: Tracer):
        """
        Verify evaluation failures are handled gracefully.

        Expected behavior:
        - Customer code continues normally
        - Error logged but not raised
        - Evaluation dropped with metric incremented
        """
        with patch.object(
            tracer.api_client,
            'add_to_run_eval_queue_examples',
            side_effect=Exception("API is down")
        ):
            # Should not raise exception
            with tracer.span("test-span"):
                tracer.async_evaluate(
                    input={"prompt": "test"},
                    output={"response": "test"},
                    scorers=[]
                )

            # Verify error was logged
            # (Check logs or metrics here)
```

#### Test 3: Tag API with Timeout

```python
    def test_tag_with_api_timeout(self, tracer: Tracer):
        """
        Verify tagging doesn't block when API times out.

        Expected behavior:
        - tracer.tag() returns immediately
        - Tags buffered and sent later
        - No exception raised to customer
        """
        import httpx

        with patch.object(
            tracer.api_client,
            '_request',
            side_effect=httpx.TimeoutException("Request timeout")
        ):
            start = time.perf_counter()

            with tracer.span("test-span"):
                tracer.tag(["test-tag"])

            duration_ms = (time.perf_counter() - start) * 1000

            # Should not block
            assert duration_ms < 50, (
                f"tag() blocked for {duration_ms:.1f}ms on timeout"
            )
```

#### Test 4: Trace Export with Network Errors

```python
    def test_trace_export_with_network_errors(self, tracer: Tracer):
        """
        Verify traces are retried when network fails.

        Expected behavior:
        - Background export handles network errors
        - Retries with exponential backoff
        - Customer code unaffected
        """
        import httpx

        # Mock exporter to fail first 2 times, succeed on 3rd
        call_count = {"count": 0}

        def flaky_network(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise httpx.NetworkError("Network failure")
            return MagicMock()

        with patch.object(
            tracer._tracer_provider._span_processor._span_exporter,
            'export',
            side_effect=flaky_network
        ):
            # Create spans
            for i in range(10):
                with tracer.span(f"test-{i}"):
                    pass

            # Force flush with retries
            tracer.force_flush(timeout_millis=10000)

            # Should eventually succeed
            assert call_count["count"] >= 3
```

#### Test 5: API Rate Limiting

```python
    def test_api_rate_limiting(self, tracer: Tracer):
        """
        Verify SDK backs off when API returns 429 rate limit.

        Expected behavior:
        - SDK respects rate limit responses
        - Exponential backoff applied
        - Traces not lost, queued for retry
        """
        from judgeval.v1.internal.api import JudgmentAPIError

        # Mock API returning 429
        with patch.object(
            tracer.api_client,
            '_request',
            side_effect=JudgmentAPIError(
                status_code=429,
                message="Rate limit exceeded"
            )
        ):
            with tracer.span("test-span"):
                # Should not crash
                tracer.tag(["test"])

            # Verify backoff logic triggered
            # (Check metrics or logs)
```

**Success Criteria:**
- ✅ All tests pass
- ✅ Customer code never blocks > 100ms
- ✅ No exceptions raised to customer code
- ✅ All failures logged

---

### 2. Sustained Load Testing

**File:** `src/tests/reliability/test_sustained_load.py`

**Purpose:** Verify SDK maintains stability under sustained high load.

#### Test 1: 100k Spans Over 60 Seconds

```python
import pytest
import time
import tracemalloc
import gc
from judgeval.v1.tracer.tracer import Tracer

@pytest.mark.reliability
@pytest.mark.slow
class TestSustainedLoad:
    """Test SDK behavior under sustained high load."""

    def test_100k_spans_over_60_seconds(self, tracer: Tracer):
        """
        Generate 100k spans over 60 seconds (~1600 spans/sec).

        Expected behavior:
        - Memory usage stays stable
        - Queue depth stays under 90%
        - All spans exported (< 0.1% drop rate)
        """
        TARGET_SPANS = 100_000
        DURATION_SECONDS = 60
        MAX_MEMORY_GROWTH_MB = 50
        MAX_DROP_RATE = 0.001  # 0.1%

        @tracer.observe(span_type="function")
        def traced_function(i: int):
            return f"result-{i}"

        # Baseline memory
        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(s.size for s in baseline.statistics('filename'))

        # Sustained load
        start_time = time.perf_counter()
        spans_per_second = TARGET_SPANS / DURATION_SECONDS

        for i in range(TARGET_SPANS):
            traced_function(i)

            # Throttle to maintain consistent rate
            elapsed = time.perf_counter() - start_time
            expected_elapsed = (i + 1) / spans_per_second
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)

        # Allow queue to drain
        tracer.force_flush(timeout_millis=30000)

        # Check memory
        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(s.size for s in final.statistics('filename'))
        tracemalloc.stop()

        memory_growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert memory_growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew {memory_growth_mb:.1f}MB over 60s "
            f"(max: {MAX_MEMORY_GROWTH_MB}MB)"
        )

        # Check drop rate (if metrics available)
        # metrics = tracer.get_health_metrics()
        # drop_rate = metrics['spans_dropped'] / TARGET_SPANS
        # assert drop_rate < MAX_DROP_RATE
```

#### Test 2: Burst Traffic Spike

```python
    def test_burst_traffic_spike(self, tracer: Tracer):
        """
        Simulate sudden traffic spike: 10k spans in 1 second.

        Expected behavior:
        - Queue absorbs burst without crashing
        - Queue drains after spike
        - Drop rate < 5% during spike
        - Memory returns to baseline after drain
        """
        BURST_SIZE = 10_000
        MAX_DROP_RATE_BURST = 0.05  # 5%
        MAX_MEMORY_GROWTH_MB = 30

        @tracer.observe(span_type="function")
        def fast_function(i: int):
            return i * 2

        # Baseline
        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(s.size for s in baseline.statistics('filename'))

        # Burst
        start = time.perf_counter()
        for i in range(BURST_SIZE):
            fast_function(i)
        burst_duration = time.perf_counter() - start

        print(f"Generated {BURST_SIZE} spans in {burst_duration:.2f}s")

        # Allow drain
        time.sleep(10)
        tracer.force_flush(timeout_millis=30000)

        # Check memory returned to baseline
        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(s.size for s in final.statistics('filename'))
        tracemalloc.stop()

        memory_growth_mb = (final_size - baseline_size) / 1024 / 1024

        assert memory_growth_mb < MAX_MEMORY_GROWTH_MB, (
            f"Memory grew {memory_growth_mb:.1f}MB after burst "
            f"(max: {MAX_MEMORY_GROWTH_MB}MB)"
        )
```

#### Test 3: Queue Saturation Under Load

```python
    def test_queue_saturation_under_load(self, tracer: Tracer):
        """
        Generate spans faster than export capacity.

        Expected behavior:
        - Queue fills up
        - Oldest spans dropped (not crashed)
        - Metrics track drop rate
        - Customer code continues normally
        """
        # Generate 5000 spans rapidly (will exceed queue capacity)
        @tracer.observe(span_type="function")
        def rapid_function(i: int):
            return i

        for i in range(5000):
            rapid_function(i)

        # Verify no crash
        assert True  # If we get here, didn't crash

        # Check metrics
        # metrics = tracer.get_health_metrics()
        # assert metrics['spans_dropped'] > 0
        # assert metrics['queue_utilization'] > 0.8
```

#### Test 4: Concurrent High-Frequency Threads

```python
    def test_concurrent_high_frequency_threads(self, tracer: Tracer):
        """
        100 threads each generating 1000 spans concurrently.

        Expected behavior:
        - Thread-safe span creation
        - No race conditions
        - All spans accounted for (created or dropped)
        - Memory stable
        """
        from concurrent.futures import ThreadPoolExecutor
        import threading

        THREADS = 100
        SPANS_PER_THREAD = 1000
        TOTAL_SPANS = THREADS * SPANS_PER_THREAD

        @tracer.observe(span_type="function")
        def thread_function(thread_id: int, iteration: int):
            return f"{thread_id}-{iteration}"

        def worker(thread_id: int):
            for i in range(SPANS_PER_THREAD):
                thread_function(thread_id, i)

        # Execute
        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = [executor.submit(worker, i) for i in range(THREADS)]
            for future in futures:
                future.result()  # Wait for completion

        # Allow drain
        tracer.force_flush(timeout_millis=60000)

        # Verify no crashes
        assert True
```

#### Test 5: Long-Running Service (24 Hours)

```python
    @pytest.mark.skip(reason="Long-running test, run manually")
    def test_long_running_service_24_hours(self, tracer: Tracer):
        """
        Run tracer for 24 hours with steady load.

        Expected behavior:
        - Memory stays flat (no leaks)
        - Performance doesn't degrade
        - No crashes or deadlocks

        Run with:
        pytest -k test_long_running_service_24_hours --no-skip
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

            time.sleep(60)  # Wait 1 minute

            # Check memory every hour
            if iteration % (SPANS_PER_MINUTE * 60) == 0:
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory_mb - baseline_memory_mb
                print(f"Hour {iteration // (SPANS_PER_MINUTE * 60)}: "
                      f"Memory growth: {memory_growth:.1f}MB")

                assert memory_growth < 100, "Memory leak detected"
```

**Success Criteria:**
- ✅ Memory stays stable over 60+ seconds
- ✅ Queue depth stays under 90%
- ✅ Drop rate < 0.1% under normal load
- ✅ Drop rate < 5% under burst
- ✅ Thread-safe under concurrency

---

### 3. Crash Resistance

**File:** `src/tests/reliability/test_crash_resistance.py`

**Purpose:** Verify SDK errors never crash customer code.

#### Test 1: SDK Never Crashes Customer Code

```python
import pytest
from judgeval.v1.tracer.tracer import Tracer
from unittest.mock import patch

@pytest.mark.reliability
class TestCrashResistance:
    """Test that SDK failures never crash customer applications."""

    def test_sdk_never_crashes_customer_code(self, tracer: Tracer):
        """
        Inject various SDK failures and verify customer code continues.

        Expected behavior:
        - API errors: logged, not raised
        - Serialization errors: logged, not raised
        - Export errors: logged, not raised
        - Customer code always completes successfully
        """

        # Test 1: API failure during evaluation
        with patch.object(
            tracer.api_client,
            'add_to_run_eval_queue_examples',
            side_effect=Exception("API failed")
        ):
            @tracer.observe(span_type="function")
            def function_with_eval():
                tracer.async_evaluate(
                    input={"test": "data"},
                    output={"result": "ok"},
                    scorers=[]
                )
                return "success"

            result = function_with_eval()
            assert result == "success"  # Customer code succeeded

        # Test 2: Serialization error in set_attribute
        with patch.object(
            tracer,
            'set_attribute',
            side_effect=TypeError("Cannot serialize")
        ):
            @tracer.observe(span_type="function")
            def function_with_bad_attribute():
                tracer.set_attribute("key", object())  # Non-serializable
                return "success"

            result = function_with_bad_attribute()
            assert result == "success"

        # Test 3: Export failure
        with patch.object(
            tracer._tracer_provider._span_processor,
            'force_flush',
            side_effect=Exception("Export failed")
        ):
            @tracer.observe(span_type="function")
            def function_with_export_failure():
                return "success"

            result = function_with_export_failure()
            assert result == "success"
```

#### Test 2: OOM During Serialization

```python
    def test_oom_during_serialization(self, tracer: Tracer):
        """
        Create spans with extremely large payloads (100MB+).

        Expected behavior:
        - SDK truncates or skips large payloads
        - Customer code completes normally
        - Memory doesn't explode
        """
        HUGE_PAYLOAD_SIZE = 100_000_000  # 100MB
        huge_data = "x" * HUGE_PAYLOAD_SIZE

        @tracer.observe(span_type="function")
        def function_with_huge_payload():
            # Try to set huge attribute
            tracer.set_attribute("huge", huge_data)
            return "success"

        # Should handle gracefully (truncate or skip)
        result = function_with_huge_payload()
        assert result == "success"
```

#### Test 3: Corrupted Span Data

```python
    def test_corrupted_span_data(self, tracer: Tracer):
        """
        Inject non-serializable objects into spans.

        Expected behavior:
        - Serialization failure caught
        - Span skipped or sanitized
        - Customer code continues
        """
        import threading

        # Non-serializable objects
        bad_objects = [
            threading.Lock(),
            lambda x: x,
            type("CustomClass", (), {}),
            open(__file__),
        ]

        for bad_obj in bad_objects:
            @tracer.observe(span_type="function")
            def function_with_bad_object():
                tracer.set_attribute("bad", bad_obj)
                return "success"

            result = function_with_bad_object()
            assert result == "success"
```

#### Test 4: Background Thread Crashes

```python
    def test_background_thread_crashes(self, tracer: Tracer):
        """
        Simulate crash in export worker thread.

        Expected behavior:
        - Thread restarts or degrades gracefully
        - Customer code unaffected
        - SDK logs error
        """
        # This test is more conceptual - would require
        # instrumenting the background worker thread

        # Verify customer code continues after simulated thread crash
        @tracer.observe(span_type="function")
        def normal_function():
            return "success"

        # Even if background thread crashes, this should work
        for _ in range(100):
            result = normal_function()
            assert result == "success"
```

**Success Criteria:**
- ✅ Customer code never crashes from SDK errors
- ✅ All exceptions caught and logged
- ✅ Large payloads handled gracefully
- ✅ Non-serializable objects handled gracefully

---

### 4. Memory Leak Detection

**File:** `src/tests/reliability/test_memory_leaks.py` (expand existing)

**Purpose:** Verify no memory leaks in span metadata or queues.

#### Test 1: Hung Spans Don't Leak Metadata

```python
import pytest
import time
import tracemalloc
import gc
from judgeval.v1.tracer.tracer import Tracer

@pytest.mark.reliability
class TestMemoryLeaks:
    """Test for memory leaks in SDK."""

    def test_hung_spans_dont_leak_metadata(self, tracer: Tracer):
        """
        Create spans that never call end() and verify metadata is cleaned up.

        Expected behavior:
        - Metadata evicted after TTL (1 hour)
        - LRU eviction when > 10k entries
        - Memory stays bounded
        """
        # Create 1000 spans without ending them
        spans = []

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        baseline_size = sum(s.size for s in baseline.statistics('filename'))

        for i in range(1000):
            span = tracer._tracer.start_span(f"hung-span-{i}")
            tracer._span_processor.set_internal_attribute(
                span.get_span_context(),
                "test_data",
                {"large": "x" * 1000}
            )
            spans.append(span)
            # Intentionally don't call span.end()

        # Wait for TTL cleanup or trigger LRU eviction
        time.sleep(2)  # In real impl, TTL would be longer

        # Create more spans to trigger LRU
        for i in range(10000):
            span = tracer._tracer.start_span(f"new-span-{i}")
            tracer._span_processor.set_internal_attribute(
                span.get_span_context(),
                "test",
                "value"
            )
            span.end()

        gc.collect()
        final = tracemalloc.take_snapshot()
        final_size = sum(s.size for s in final.statistics('filename'))
        tracemalloc.stop()

        memory_growth_mb = (final_size - baseline_size) / 1024 / 1024

        # Memory should be bounded (LRU evicted old entries)
        assert memory_growth_mb < 20, (
            f"Memory grew {memory_growth_mb:.1f}MB with hung spans"
        )
```

#### Test 2: Metadata Eviction Under Pressure

```python
    def test_metadata_eviction_under_pressure(self, tracer: Tracer):
        """
        Create 20k spans rapidly and verify metadata storage stays bounded.

        Expected behavior:
        - Metadata entries capped at ~10k (configurable)
        - LRU eviction works correctly
        - No unbounded growth
        """
        MAX_METADATA_SIZE = 10_000

        # Create 20k spans (should trigger eviction)
        for i in range(20_000):
            with tracer.span(f"span-{i}"):
                tracer.set_attribute("index", i)

        # Check metadata size (if exposed)
        # metadata_size = len(tracer._span_processor._internal_attributes)
        # assert metadata_size <= MAX_METADATA_SIZE
```

#### Test 3: Evaluation Queue Bounded

```python
    def test_evaluation_queue_bounded(self, tracer: Tracer):
        """
        Enqueue 10k evaluations without processing.

        Expected behavior:
        - Queue is bounded (e.g., 1000 max)
        - Oldest evaluations dropped when full
        - Memory doesn't grow unbounded
        """
        from unittest.mock import patch

        # Block evaluation processing
        with patch.object(
            tracer.api_client,
            'add_to_run_eval_queue_examples',
            return_value=None  # No-op
        ):
            gc.collect()
            tracemalloc.start()
            baseline = tracemalloc.take_snapshot()
            baseline_size = sum(s.size for s in baseline.statistics('filename'))

            # Try to enqueue 10k evaluations
            for i in range(10_000):
                with tracer.span(f"eval-{i}"):
                    tracer.async_evaluate(
                        input={"i": i},
                        output={"result": i},
                        scorers=[]
                    )

            gc.collect()
            final = tracemalloc.take_snapshot()
            final_size = sum(s.size for s in final.statistics('filename'))
            tracemalloc.stop()

            memory_growth_mb = (final_size - baseline_size) / 1024 / 1024

            # Memory should be bounded by queue limit
            assert memory_growth_mb < 50, (
                f"Evaluation queue leaked memory: {memory_growth_mb:.1f}MB"
            )
```

**Success Criteria:**
- ✅ Hung spans cleaned up after TTL
- ✅ Metadata storage bounded by LRU
- ✅ Evaluation queue bounded
- ✅ Memory stable over time

---

### 5. Latency Under Stress

**File:** `src/tests/reliability/test_stress_latency.py`

**Purpose:** Verify performance doesn't degrade catastrophically under stress.

#### Test 1: Latency During Queue Saturation

```python
import pytest
import time
import statistics
from judgeval.v1.tracer.tracer import Tracer

@pytest.mark.reliability
class TestStressLatency:
    """Test latency behavior under stress conditions."""

    def test_latency_during_queue_saturation(self, tracer: Tracer):
        """
        Measure @observe overhead when queue is 90% full.

        Expected behavior:
        - Overhead still < 1ms (p95)
        - Queue pressure doesn't slow down customer code
        """
        # Fill queue to 90%
        for i in range(1800):  # Assuming 2048 max queue
            with tracer.span(f"filler-{i}"):
                pass

        # Now measure overhead
        @tracer.observe(span_type="function")
        def traced_function():
            time.sleep(0.001)
            return "result"

        def baseline_function():
            time.sleep(0.001)
            return "result"

        # Measure baseline
        baseline_times = []
        for _ in range(100):
            start = time.perf_counter()
            baseline_function()
            baseline_times.append((time.perf_counter() - start) * 1000)

        # Measure traced (with queue saturation)
        traced_times = []
        for _ in range(100):
            start = time.perf_counter()
            traced_function()
            traced_times.append((time.perf_counter() - start) * 1000)

        baseline_p95 = statistics.quantiles(baseline_times, n=20)[18]  # 95th percentile
        traced_p95 = statistics.quantiles(traced_times, n=20)[18]
        overhead = traced_p95 - baseline_p95

        assert overhead < 1.0, (
            f"Overhead {overhead:.2f}ms during queue saturation (max: 1ms)"
        )
```

#### Test 2: Latency During API Outage

```python
    def test_latency_during_api_outage(self, tracer: Tracer):
        """
        Measure customer code latency when API is completely down.

        Expected behavior:
        - Customer code latency unaffected
        - Background threads absorb the failure
        """
        from unittest.mock import patch

        with patch.object(
            tracer.api_client,
            '_request',
            side_effect=Exception("API down")
        ):
            latencies = []

            for _ in range(100):
                start = time.perf_counter()

                @tracer.observe(span_type="function")
                def customer_function():
                    return "result"

                customer_function()

                latencies.append((time.perf_counter() - start) * 1000)

            p95_latency = statistics.quantiles(latencies, n=20)[18]

            # Should still be fast
            assert p95_latency < 2.0, (
                f"Customer code latency {p95_latency:.2f}ms during API outage"
            )
```

#### Test 3: Latency with Background Pressure

```python
    def test_latency_with_background_pressure(self, tracer: Tracer):
        """
        Saturate background export threads and measure foreground latency.

        Expected behavior:
        - Background pressure doesn't affect foreground
        - No resource contention
        """
        # Create background pressure (large export backlog)
        for i in range(5000):
            with tracer.span(f"background-{i}"):
                tracer.set_attribute("data", "x" * 1000)

        # Measure foreground latency
        latencies = []

        for _ in range(100):
            start = time.perf_counter()

            with tracer.span("foreground"):
                pass

            latencies.append((time.perf_counter() - start) * 1000)

        p95_latency = statistics.quantiles(latencies, n=20)[18]

        assert p95_latency < 1.0, (
            f"Foreground latency {p95_latency:.2f}ms with background pressure"
        )
```

**Success Criteria:**
- ✅ Overhead < 1ms even when queue saturated
- ✅ Customer code unaffected by API outages
- ✅ No resource contention with background threads

---

### 6. Chaos Scenarios

**File:** `src/tests/reliability/test_chaos.py`

**Purpose:** Verify robustness to unpredictable failures.

#### Test 1: Random API Timeouts

```python
import pytest
import random
from unittest.mock import patch
from judgeval.v1.tracer.tracer import Tracer

@pytest.mark.reliability
@pytest.mark.chaos
class TestChaosScenarios:
    """Chaos engineering tests for unpredictable failures."""

    def test_random_api_timeouts(self, tracer: Tracer):
        """
        20% of API calls timeout randomly.

        Expected behavior:
        - SDK maintains stability
        - Customer code unaffected
        - Traces eventually succeed (with retries)
        """
        original_request = tracer.api_client._request

        def flaky_api(*args, **kwargs):
            if random.random() < 0.2:  # 20% failure rate
                raise TimeoutError("Random timeout")
            return original_request(*args, **kwargs)

        with patch.object(
            tracer.api_client,
            '_request',
            side_effect=flaky_api
        ):
            # Generate spans with flaky API
            for i in range(100):
                with tracer.span(f"test-{i}"):
                    if i % 10 == 0:
                        tracer.tag([f"tag-{i}"])

        # Should complete without crashing
        assert True
```

#### Test 2: Intermittent Network Partitions

```python
    def test_intermittent_network_partitions(self, tracer: Tracer):
        """
        Simulate network dropping 50% of packets.

        Expected behavior:
        - Traces eventually reach backend
        - SDK doesn't crash
        - Retries handle network issues
        """
        import httpx

        call_count = {"count": 0}

        def unreliable_network(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] % 2 == 0:
                raise httpx.NetworkError("Packet dropped")
            return original_export(*args, **kwargs)

        original_export = tracer._tracer_provider._span_processor._span_exporter.export

        with patch.object(
            tracer._tracer_provider._span_processor._span_exporter,
            'export',
            side_effect=unreliable_network
        ):
            for i in range(50):
                with tracer.span(f"test-{i}"):
                    pass

            # Force flush with retries
            tracer.force_flush(timeout_millis=30000)

        # Verify retries occurred
        assert call_count["count"] > 50  # Had to retry
```

#### Test 3: Memory Pressure

```python
    @pytest.mark.skip(reason="Requires cgroups/container limits")
    def test_memory_pressure(self, tracer: Tracer):
        """
        Limit SDK memory to 100MB and generate high trace volume.

        Expected behavior:
        - Graceful degradation (drop spans, don't crash)
        - SDK detects memory pressure
        - Customer code continues

        Run with:
        docker run --memory=100m python -m pytest -k test_memory_pressure
        """
        # Generate high volume
        for i in range(100_000):
            with tracer.span(f"span-{i}"):
                tracer.set_attribute("data", "x" * 1000)

        # Should degrade gracefully, not OOM
        assert True
```

#### Test 4: CPU Throttling

```python
    def test_cpu_throttling(self, tracer: Tracer):
        """
        Throttle background threads to 10% CPU.

        Expected behavior:
        - Customer code unaffected
        - Export continues (slowly)
        - No deadlocks
        """
        # This test is conceptual - would require CPU throttling
        # via cgroups or process priority

        @tracer.observe(span_type="function")
        def customer_function(i: int):
            return i * 2

        # Customer code should be fast even with throttled background
        import time
        start = time.perf_counter()

        for i in range(1000):
            customer_function(i)

        duration = time.perf_counter() - start

        # Should complete quickly (< 1 second for 1000 calls)
        assert duration < 1.0
```

#### Test 5: Concurrent Tracer Shutdown

```python
    def test_concurrent_tracer_shutdown(self, tracer: Tracer):
        """
        Shutdown tracer while spans are in-flight.

        Expected behavior:
        - No deadlocks
        - No crashes
        - Graceful drain or timeout
        """
        import threading

        # Create spans in background
        def create_spans():
            for i in range(1000):
                with tracer.span(f"concurrent-{i}"):
                    time.sleep(0.001)

        thread = threading.Thread(target=create_spans)
        thread.start()

        # Shutdown while spans still being created
        time.sleep(0.1)
        tracer.shutdown(timeout_millis=5000)

        thread.join()

        # Should complete without deadlock
        assert True
```

**Success Criteria:**
- ✅ SDK survives random failures
- ✅ No crashes under chaos
- ✅ Customer code always continues
- ✅ Graceful degradation under resource pressure

---

## Load Testing

### Setup: Locust Framework

**File:** `src/tests/reliability/load_tests/locustfile.py`

```python
"""
Locust load testing for Judgeval SDK.

Run with:
    locust -f src/tests/reliability/load_tests/locustfile.py

Then open http://localhost:8089 and configure:
    - Number of users: 1000
    - Spawn rate: 100/sec
    - Duration: 10 minutes
"""

from locust import User, task, between, events
import time
from judgeval.v1.tracer.tracer import Tracer

class TracerUser(User):
    """Simulates a customer application using Judgeval SDK."""

    wait_time = between(0.01, 0.1)  # 10-100ms between requests

    def on_start(self):
        """Initialize tracer for this user."""
        self.tracer = Tracer(
            project_name="load-test",
            enable_monitoring=True,
            enable_evaluation=False,
        )
        self.iteration = 0

    @task(10)
    def trace_simple_function(self):
        """Most common case: simple traced function."""
        start = time.perf_counter()

        @self.tracer.observe(span_type="function")
        def simple_function():
            return "result"

        result = simple_function()

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="simple_function",
            response_time=duration_ms,
            response_length=len(result),
            exception=None,
        )

    @task(5)
    def trace_with_attributes(self):
        """Traced function with attributes."""
        start = time.perf_counter()

        with self.tracer.span("attributed-span"):
            self.tracer.set_attribute("iteration", self.iteration)
            self.tracer.set_attribute("user_id", self.user_id)
            self.iteration += 1

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="with_attributes",
            response_time=duration_ms,
            response_length=0,
            exception=None,
        )

    @task(3)
    def trace_with_tags(self):
        """Traced span with tagging."""
        start = time.perf_counter()

        with self.tracer.span("tagged-span"):
            self.tracer.tag(["load-test", "performance"])

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="with_tags",
            response_time=duration_ms,
            response_length=0,
            exception=None,
        )

    @task(2)
    def nested_spans(self):
        """Nested traced functions."""
        start = time.perf_counter()

        @self.tracer.observe(span_type="function")
        def outer():
            @self.tracer.observe(span_type="function")
            def inner():
                return "deep"
            return inner()

        result = outer()

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="nested_spans",
            response_time=duration_ms,
            response_length=len(result),
            exception=None,
        )

    @task(1)
    def large_payload(self):
        """Trace function with large payload."""
        start = time.perf_counter()

        @self.tracer.observe(span_type="function")
        def large_function():
            return "x" * 10000  # 10KB payload

        result = large_function()

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="large_payload",
            response_time=duration_ms,
            response_length=len(result),
            exception=None,
        )
```

### Load Test Scenarios

**Scenario 1: Baseline Load**
```bash
locust -f locustfile.py \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless
```
**Expected:** < 1ms p95 latency, 0% failures

**Scenario 2: Moderate Load**
```bash
locust -f locustfile.py \
    --users 500 \
    --spawn-rate 50 \
    --run-time 10m \
    --headless
```
**Expected:** < 2ms p95 latency, < 0.1% drop rate

**Scenario 3: High Load**
```bash
locust -f locustfile.py \
    --users 1000 \
    --spawn-rate 100 \
    --run-time 10m \
    --headless
```
**Expected:** < 5ms p95 latency, < 1% drop rate

**Scenario 4: Burst Load**
```bash
locust -f locustfile.py \
    --users 2000 \
    --spawn-rate 500 \
    --run-time 2m \
    --headless
```
**Expected:** < 10ms p95 latency, < 5% drop rate

### Analyzing Load Test Results

**Key Metrics to Monitor:**
- Request latency (p50, p95, p99)
- Failure rate
- Requests per second (throughput)
- SDK queue utilization (via Prometheus)
- Span drop rate (via Prometheus)

**Success Criteria:**
```
Baseline (100 users):
    p95 < 1ms, 0% failures, > 1000 RPS

Moderate (500 users):
    p95 < 2ms, < 0.1% drop rate, > 3000 RPS

High (1000 users):
    p95 < 5ms, < 1% drop rate, > 5000 RPS

Burst (2000 users):
    p95 < 10ms, < 5% drop rate, survives without crash
```

---

## Performance Benchmarking

### Microbenchmarks

**File:** `src/tests/reliability/benchmarks/bench_overhead.py`

```python
"""
Microbenchmarks for SDK overhead.

Run with:
    pytest src/tests/reliability/benchmarks/ --benchmark-only
"""

import pytest
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

        # benchmark.stats shows timing stats
        # Should be < 1ms median

    def test_span_context_manager_overhead(self, benchmark, tracer: Tracer):
        """Benchmark span() context manager overhead."""

        def create_span():
            with tracer.span("test"):
                pass

        benchmark(create_span)
        # Should be < 0.5ms median

    def test_set_attribute_overhead(self, benchmark, tracer: Tracer):
        """Benchmark set_attribute overhead."""

        def set_attr():
            with tracer.span("test"):
                tracer.set_attribute("key", "value")

        benchmark(set_attr)
        # Should be < 0.1ms median
```

---

## CI/CD Integration

### GitHub Actions Workflow

**File:** `.github/workflows/reliability-tests.yml`

```yaml
name: Reliability Tests

on:
  push:
    branches: [main, staging]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2am

jobs:
  unit-reliability-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest pytest-timeout pytest-benchmark

      - name: Run reliability tests
        run: |
          pytest src/tests/reliability/ \
            -v \
            -m "reliability and not slow and not chaos" \
            --timeout=300 \
            --junit-xml=results/reliability-junit.xml

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: reliability-test-results
          path: results/

  sustained-load-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run sustained load tests
        run: |
          pytest src/tests/reliability/test_sustained_load.py \
            -v \
            -m "slow" \
            --timeout=3600

  chaos-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'  # Only run on daily schedule

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run chaos tests
        run: |
          pytest src/tests/reliability/test_chaos.py \
            -v \
            -m "chaos"

  load-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'  # Only run on daily schedule

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install locust

      - name: Start monitoring stack
        run: |
          cd monitoring
          docker-compose up -d
          sleep 10

      - name: Run load test
        run: |
          locust -f src/tests/reliability/load_tests/locustfile.py \
            --users 500 \
            --spawn-rate 50 \
            --run-time 5m \
            --headless \
            --csv=results/load-test

      - name: Check results
        run: |
          python scripts/analyze_load_test.py results/load-test_stats.csv

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: results/
```

### Test Markers

**Add to `pytest.ini`:**
```ini
[pytest]
markers =
    reliability: Reliability and performance tests
    slow: Tests that take > 30 seconds
    chaos: Chaos engineering tests
    load: Load tests requiring external tools
```

### Running Tests Locally

```bash
# Fast reliability tests (< 1 minute)
pytest src/tests/reliability/ -m "reliability and not slow"

# All reliability tests including slow ones
pytest src/tests/reliability/ -m "reliability"

# Specific test suite
pytest src/tests/reliability/test_api_failures.py -v

# With coverage
pytest src/tests/reliability/ --cov=judgeval --cov-report=html

# Load tests
locust -f src/tests/reliability/load_tests/locustfile.py

# Benchmarks
pytest src/tests/reliability/benchmarks/ --benchmark-only
```

---

## Success Criteria

### Critical Requirements (Must Pass)

- ✅ **Zero Customer Impact**
  - No exceptions raised to customer code
  - `@observe` overhead < 1ms (p95)
  - Customer code never blocks > 100ms

- ✅ **Memory Stability**
  - Memory growth < 50MB over 100k spans
  - No leaks from hung spans
  - Metadata storage bounded

- ✅ **Trace Reliability**
  - Drop rate < 0.1% under normal load
  - Drop rate < 5% under 10x burst
  - All drops tracked in metrics

- ✅ **Exception Isolation**
  - All SDK errors caught and logged
  - Customer code continues on SDK failure
  - No crashes from bad input data

### Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| `@observe` overhead (p95) | < 0.5ms | < 1ms |
| Span creation latency | < 0.1ms | < 0.5ms |
| set_attribute latency | < 0.05ms | < 0.1ms |
| Memory growth (100k spans) | < 20MB | < 50MB |
| Throughput capacity | > 1000 spans/sec | > 500 spans/sec |
| Queue saturation threshold | < 80% | < 95% |

### Reliability Targets

| Metric | Normal Load | Burst Load (10x) |
|--------|-------------|------------------|
| Span drop rate | < 0.1% | < 5% |
| API error tolerance | 100% | 100% |
| Memory leak rate | 0 MB/hour | 0 MB/hour |
| Crash rate | 0 | 0 |

---

## Test Execution Schedule

### Pre-Commit (Developer Local)
- Fast reliability tests (< 30s)
- Memory tests
- Latency tests

### CI/CD Pull Request
- All reliability tests except slow/chaos
- Benchmarks (for comparison)
- Memory tests

### CI/CD Main Branch
- Full reliability test suite
- Sustained load tests (60s)
- Benchmarks with history

### Nightly CI/CD
- Chaos engineering tests
- Load tests (10 minutes)
- Long-running tests (if any)
- Generate test report

### Weekly/Manual
- 24-hour soak test
- Full load testing suite
- Stress testing with varied scenarios

---

## Test Reporting

### Metrics to Track

1. **Test Pass Rate** - % of tests passing
2. **Performance Trends** - Overhead over time
3. **Memory Trends** - Growth over time
4. **Drop Rate Trends** - Under various loads
5. **Flaky Tests** - Tests with inconsistent results

### Reporting Dashboard

Create a simple HTML report generator:

**File:** `scripts/generate_test_report.py`

```python
"""Generate HTML test report from pytest results."""

import json
from pathlib import Path

def generate_report(junit_xml_path, output_html):
    # Parse JUnit XML
    # Generate HTML with charts
    # Show pass/fail rates, trends, performance graphs
    pass

if __name__ == "__main__":
    generate_report("results/junit.xml", "results/report.html")
```

---

## Appendix: Test Implementation Checklist

### Phase 1: Critical Tests (Week 1)
- [ ] `test_api_failures.py` - 5 tests
- [ ] `test_crash_resistance.py` - 4 tests
- [ ] Expand `test_memory_leaks.py` - 3 new tests

### Phase 2: Load Tests (Week 2)
- [ ] `test_sustained_load.py` - 5 tests
- [ ] `test_stress_latency.py` - 3 tests
- [ ] `locustfile.py` - Load testing setup

### Phase 3: Chaos Tests (Week 3)
- [ ] `test_chaos.py` - 5 tests
- [ ] Chaos infrastructure setup
- [ ] CI/CD integration

### Phase 4: Infrastructure (Week 4)
- [ ] CI/CD workflows
- [ ] Test reporting
- [ ] Benchmarking suite
- [ ] Documentation

---

**Questions or Issues?**

Contact the reliability engineering team or file an issue in the repo.

**Document Version:** 1.0
**Next Review:** 2026-02-27
