# Judgeval SDK Reliability Analysis & Test Plan

**Date:** 2026-01-27
**Scope:** Tracer system reliability, performance impact, and monitoring
**Goal:** Ensure the SDK never negatively impacts customer applications

---

## Executive Summary

The Judgeval SDK has a generally well-architected tracing system with good thread safety and exception isolation. However, **4 critical issues** were identified that could negatively impact customers under production load:

1. **Blocking synchronous API calls** in the critical path (evaluation enqueue, tagging)
2. **Potential memory leak** in span metadata storage for hung spans
3. **Silent trace dropping** under sustained load without alerting
4. **No observability** into SDK health (dropped spans, API failures, queue saturation)

This document details each problem, proposes solutions, and outlines a comprehensive testing and monitoring strategy.

---

## Part 1: Problems Found & Proposed Solutions

### Critical Priority Issues

#### 🚨 Problem 1: Blocking Evaluation Enqueue on Critical Path

**Location:** [base_tracer.py:412-416](src/judgeval/v1/tracer/base_tracer.py#L412-L416)

**Description:**
```python
def _enqueue_evaluation(self, evaluation_run: ExampleEvaluationRun) -> None:
    try:
        self.api_client.add_to_run_eval_queue_examples(evaluation_run)  # BLOCKING!
    except Exception as e:
        judgeval_logger.error(f"Failed to enqueue evaluation run: {e}")
```

- Uses synchronous `httpx.Client` with 30-second timeout
- Called from `async_evaluate()` which runs in customer code context
- If API is slow/down, blocks calling thread for up to 30 seconds
- Happens when `tracer.async_evaluate()` is called within a span

**Impact:**
- High-throughput customer code could experience 30s hangs
- Async event loops blocked by sync I/O
- Cascading timeouts in customer applications

**Proposed Solution:**

**Option A: Background Thread Queue (Recommended)**
```python
import queue
import threading

class BaseTracer:
    def __init__(self, ...):
        self._eval_queue = queue.Queue(maxsize=1000)
        self._eval_worker = threading.Thread(target=self._eval_worker_loop, daemon=True)
        self._eval_worker.start()

    def _enqueue_evaluation(self, evaluation_run: ExampleEvaluationRun) -> None:
        try:
            self._eval_queue.put_nowait(evaluation_run)
        except queue.Full:
            judgeval_logger.warning("Evaluation queue full, dropping eval")
            # Increment metric: evaluations_dropped_total

    def _eval_worker_loop(self):
        while True:
            try:
                eval_run = self._eval_queue.get(timeout=1.0)
                self.api_client.add_to_run_eval_queue_examples(eval_run)
            except queue.Empty:
                continue
            except Exception as e:
                judgeval_logger.error(f"Failed to send evaluation: {e}")
```

**Benefits:**
- Non-blocking for customer code
- Bounded queue prevents memory issues
- Graceful degradation under load

**Option B: Native Async (More invasive)**
- Add `httpx.AsyncClient` alongside sync client
- Make `_enqueue_evaluation` async
- Requires larger refactor

**Recommendation:** Implement Option A immediately (low-risk, high-impact fix).

---

#### 🚨 Problem 2: Blocking Tag API Calls

**Location:** [base_tracer.py:335-351](src/judgeval/v1/tracer/base_tracer.py#L335-L351)

**Description:**
```python
def tag(self, tags: str | list[str]) -> None:
    # ...
    self.api_client._request(
        "POST",
        url_for("/traces/tags/add", self.api_client.base_url),
        {...}
    )
```

- Direct synchronous HTTP POST on every `tracer.tag()` call
- Called from customer code (not buffered)
- 30-second timeout applies

**Impact:**
- Customers calling `tracer.tag()` in hot paths will experience latency
- High-frequency tagging causes API rate limiting
- Network issues cause customer code to hang

**Proposed Solution:**

**Batch Tag Buffer System**
```python
from collections import defaultdict
from threading import Lock, Thread
import time

class BaseTracer:
    def __init__(self, ...):
        self._tag_buffer: defaultdict[str, set[str]] = defaultdict(set)
        self._tag_lock = Lock()
        self._tag_flush_thread = Thread(target=self._tag_flush_loop, daemon=True)
        self._tag_flush_thread.start()

    def tag(self, tags: str | list[str]) -> None:
        span_context = self._get_sampled_span_context()
        if span_context is None:
            return

        trace_id = format(span_context.trace_id, "032x")
        tags_list = tags if isinstance(tags, list) else [tags]

        with self._tag_lock:
            self._tag_buffer[trace_id].update(tags_list)

    def _tag_flush_loop(self):
        """Flush tags every 5 seconds or when buffer reaches 100 traces."""
        while True:
            time.sleep(5)
            self._flush_tags()

    def _flush_tags(self):
        with self._tag_lock:
            if not self._tag_buffer:
                return
            buffer_copy = dict(self._tag_buffer)
            self._tag_buffer.clear()

        # Send batched request
        try:
            self.api_client._request(
                "POST",
                url_for("/traces/tags/add_batch", self.api_client.base_url),
                {
                    "project_name": self.project_name,
                    "tags_by_trace": {
                        trace_id: list(tags)
                        for trace_id, tags in buffer_copy.items()
                    }
                }
            )
        except Exception as e:
            judgeval_logger.error(f"Failed to flush tags: {e}")
```

**Benefits:**
- Non-blocking customer code
- Reduces API calls by batching
- Deduplicates tags per trace
- Graceful failure handling

**Backend Change Required:** Add `/traces/tags/add_batch` endpoint

---

#### 🚨 Problem 3: Memory Leak in Span Metadata

**Location:** [judgment_span_processor.py:49-66](src/judgeval/v1/tracer/processors/judgment_span_processor.py#L49-L66)

**Description:**
```python
self._internal_attributes: defaultdict[tuple[int, int], dict[str, Any]] = defaultdict(dict)
```

- Stores metadata keyed by `(trace_id, span_id)`
- Only cleaned up when `_cleanup_span_state()` is called on span end
- If span never completes (hung, exception, memory corruption), metadata persists forever
- Unbounded growth over long-running applications

**Impact:**
- Memory leak in long-running services
- Grows linearly with number of hung spans
- Each span can store arbitrary-sized metadata

**Proposed Solution:**

**TTL-Based Eviction with LRU Fallback**
```python
import time
from collections import OrderedDict

class JudgmentSpanProcessor(BatchSpanProcessor):
    MAX_METADATA_SIZE = 10_000  # Maximum entries
    METADATA_TTL_SECONDS = 3600  # 1 hour TTL

    def __init__(self, ...):
        super().__init__(...)
        self._internal_attributes: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self._metadata_timestamps: dict[tuple[int, int], float] = {}
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def set_internal_attribute(self, span_context: SpanContext, key: str, value: Any) -> None:
        span_key = self._get_span_key(span_context)
        self._internal_attributes[span_key][key] = value
        self._metadata_timestamps[span_key] = time.time()

        # LRU eviction if too large
        if len(self._internal_attributes) > self.MAX_METADATA_SIZE:
            oldest_key = next(iter(self._internal_attributes))
            self._internal_attributes.pop(oldest_key, None)
            self._metadata_timestamps.pop(oldest_key, None)
            judgeval_logger.warning(f"Evicted span metadata (LRU): {oldest_key}")

    def _cleanup_loop(self):
        """Periodically clean up expired metadata."""
        while True:
            time.sleep(60)  # Run every minute
            self._cleanup_expired()

    def _cleanup_expired(self):
        now = time.time()
        expired_keys = [
            key for key, ts in self._metadata_timestamps.items()
            if now - ts > self.METADATA_TTL_SECONDS
        ]

        for key in expired_keys:
            self._internal_attributes.pop(key, None)
            self._metadata_timestamps.pop(key, None)

        if expired_keys:
            judgeval_logger.info(f"Cleaned up {len(expired_keys)} expired span metadata entries")
```

**Benefits:**
- Bounded memory usage via LRU eviction
- TTL ensures hung spans don't leak memory
- Graceful degradation under load

---

#### 🚨 Problem 4: Silent Trace Dropping Under Load

**Location:** OpenTelemetry `BatchSpanProcessor` (default max_queue_size: 2048)

**Description:**
- Default queue size: 2048 spans
- At 5-second batch interval: ~410 spans/second capacity
- High-frequency tracing (10k/sec in tests) overflows queue
- Overflow behavior: **Oldest spans silently dropped (FIFO)**
- No metrics, no alerts, no customer visibility

**Impact:**
- Production trace loss during traffic spikes
- Customers unaware they're missing data
- Debugging issues with incomplete traces

**Proposed Solution:**

**Add SDK Metrics & Queue Monitoring**
```python
from dataclasses import dataclass
import time

@dataclass
class TracerMetrics:
    """SDK health metrics."""
    spans_created: int = 0
    spans_exported: int = 0
    spans_dropped: int = 0  # Queue full
    spans_failed: int = 0   # Export failed
    evaluations_enqueued: int = 0
    evaluations_dropped: int = 0
    tags_buffered: int = 0
    tags_flushed: int = 0
    api_errors: int = 0
    last_export_timestamp: float = 0
    last_export_duration_ms: float = 0

class JudgmentSpanProcessor(BatchSpanProcessor):
    def __init__(self, ...):
        super().__init__(...)
        self.metrics = TracerMetrics()
        self._metrics_callback = None  # Hook for external monitoring

    def on_start(self, span, parent_context):
        super().on_start(span, parent_context)
        self.metrics.spans_created += 1

    def on_end(self, span):
        # Check queue size before adding
        if self._is_queue_full():
            self.metrics.spans_dropped += 1
            judgeval_logger.warning(
                f"Span queue full ({self.max_queue_size}), dropping span"
            )
            return

        super().on_end(span)

    def _is_queue_full(self):
        # Access internal queue size (requires minor BatchSpanProcessor modification)
        return self.queue.qsize() >= self.max_queue_size * 0.9  # 90% threshold

    def export(self, spans):
        start = time.perf_counter()
        try:
            result = super().export(spans)
            self.metrics.spans_exported += len(spans)
            self.metrics.last_export_timestamp = time.time()
            self.metrics.last_export_duration_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as e:
            self.metrics.spans_failed += len(spans)
            self.metrics.api_errors += 1
            raise

    def get_metrics(self) -> TracerMetrics:
        """Expose metrics for external monitoring."""
        return self.metrics
```

**Add Metrics Endpoint:**
```python
class BaseTracer:
    def get_health_metrics(self) -> dict:
        """Return SDK health metrics for monitoring."""
        processor = self._get_span_processor()
        metrics = processor.get_metrics()

        return {
            "spans_created": metrics.spans_created,
            "spans_exported": metrics.spans_exported,
            "spans_dropped": metrics.spans_dropped,
            "spans_failed": metrics.spans_failed,
            "drop_rate": metrics.spans_dropped / max(metrics.spans_created, 1),
            "queue_utilization": processor.queue.qsize() / processor.max_queue_size,
            "last_export_age_seconds": time.time() - metrics.last_export_timestamp,
            "evaluations_dropped": metrics.evaluations_dropped,
            "api_errors": metrics.api_errors,
        }
```

**Benefits:**
- Visibility into SDK health
- Alert on high drop rates
- Diagnose production issues
- Prometheus integration ready

---

### Medium Priority Issues

#### ⚠️ Problem 5: No Retry Logic on API Failures

**Location:** [api/__init__.py:43-60](src/judgeval/v1/internal/api/__init__.py#L43-L60)

**Description:**
- All API requests fail immediately on timeout/error
- No exponential backoff or retry
- Transient network issues cause permanent trace loss

**Proposed Solution:**

**Add Exponential Backoff with Jitter**
```python
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class JudgmentAPIClient:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True,
    )
    def _request_with_retry(self, method, url, payload, params=None):
        """Request with exponential backoff retry."""
        return self._request(method, url, payload, params)
```

**Benefits:**
- Resilience to transient failures
- Reduces trace loss
- Configurable retry policy

---

#### ⚠️ Problem 6: Default Queue Size May Be Insufficient

**Location:** OpenTelemetry BatchSpanProcessor defaults

**Description:**
- Default `max_queue_size: 2048` spans
- Supports ~410 spans/second with 5-second batch interval
- Test suite expects 10,000+ spans/second (`test_high_frequency_tracing`)

**Proposed Solution:**

**Make Queue Size Configurable with Higher Defaults**
```python
class TracerFactory:
    def create(
        self,
        project_name: str,
        max_queue_size: int = 8192,  # Increased from 2048
        max_export_batch_size: int = 512,
        schedule_delay_millis: int = 5000,
        export_timeout_millis: int = 30000,
        ...
    ):
        processor = JudgmentSpanProcessor(
            exporter,
            max_queue_size=max_queue_size,
            max_export_batch_size=max_export_batch_size,
            schedule_delay_millis=schedule_delay_millis,
            export_timeout_millis=export_timeout_millis,
        )
```

**Benefits:**
- Higher throughput capacity
- Customer configurability
- Backward compatible

---

## Part 2: Comprehensive Test Plan

### 2.1 New Reliability Tests Needed

#### Test Suite 1: API Failure Resilience
**File:** `src/tests/reliability/test_api_failures.py`

Tests to add:
1. **test_evaluation_enqueue_with_slow_api**
   - Mock API with 10-second delay
   - Verify customer code doesn't block
   - Verify evaluation still succeeds eventually

2. **test_evaluation_enqueue_with_api_down**
   - Mock API returning 500 errors
   - Verify customer code continues normally
   - Verify evaluations are dropped gracefully with logging

3. **test_tag_with_api_timeout**
   - Mock tag API with timeouts
   - Verify tagging doesn't block customer code
   - Verify tags are buffered and retried

4. **test_trace_export_with_network_errors**
   - Simulate intermittent network failures
   - Verify traces are retried
   - Verify eventual consistency

5. **test_api_rate_limiting**
   - Mock API returning 429 (rate limit)
   - Verify SDK backs off appropriately
   - Verify traces not lost

#### Test Suite 2: Sustained Load Testing
**File:** `src/tests/reliability/test_sustained_load.py`

Tests to add:
1. **test_100k_spans_over_60_seconds**
   - Sustained high-frequency tracing
   - Monitor memory usage (should be stable)
   - Monitor queue depth (should not saturate)
   - Verify all spans exported

2. **test_burst_traffic_spike**
   - Simulate 10k spans in 1 second, then idle
   - Verify queue drains without dropping spans
   - Verify memory returns to baseline

3. **test_queue_saturation_under_load**
   - Generate spans faster than export capacity
   - Verify spans are dropped gracefully (not crashed)
   - Verify metrics track drop rate

4. **test_concurrent_high_frequency_threads**
   - 100 threads each generating 1000 spans
   - Verify thread safety
   - Verify no race conditions
   - Verify all spans accounted for

5. **test_long_running_service_24_hours** (optional, CI skip)
   - Run tracer for 24 hours with steady load
   - Monitor memory (should be flat)
   - Verify no leaks or degradation

#### Test Suite 3: Crash Resistance
**File:** `src/tests/reliability/test_crash_resistance.py`

Tests to add:
1. **test_sdk_never_crashes_customer_code**
   - Inject various SDK failures (API errors, serialization errors, etc.)
   - Verify customer code continues normally
   - Verify exceptions are logged, not raised

2. **test_oom_during_serialization**
   - Create spans with extremely large payloads (100MB+)
   - Verify SDK handles gracefully (truncate or skip)
   - Verify customer code not affected

3. **test_corrupted_span_data**
   - Inject non-serializable objects into spans
   - Verify serialization failure is caught
   - Verify span is skipped, not crashed

4. **test_background_thread_crashes**
   - Simulate crash in export worker thread
   - Verify thread restarts or degrades gracefully
   - Verify customer code unaffected

#### Test Suite 4: Memory Leak Detection
**File:** `src/tests/reliability/test_memory_leaks.py` (expand existing)

Tests to add:
1. **test_hung_spans_dont_leak_metadata**
   - Create spans that never call `end()`
   - Verify metadata is evicted after TTL
   - Verify memory doesn't grow unbounded

2. **test_metadata_eviction_under_pressure**
   - Create 20k spans rapidly
   - Verify metadata storage stays bounded (< 10k entries)
   - Verify LRU eviction works

3. **test_evaluation_queue_bounded**
   - Enqueue 10k evaluations without processing
   - Verify queue is bounded
   - Verify oldest evaluations dropped

#### Test Suite 5: Latency Under Stress
**File:** `src/tests/reliability/test_stress_latency.py`

Tests to add:
1. **test_latency_during_queue_saturation**
   - Fill queue to 90% capacity
   - Measure `@observe` overhead
   - Verify overhead still < 1ms

2. **test_latency_during_api_outage**
   - Mock API as down
   - Verify customer code latency unaffected
   - Background threads should absorb the failure

3. **test_latency_with_background_pressure**
   - Saturate background export threads
   - Measure foreground `@observe` latency
   - Verify no cross-contamination

---

### 2.2 Load Testing Infrastructure

#### Setup: Locust Load Testing Framework
**File:** `src/tests/reliability/load_tests/locustfile.py`

```python
from locust import User, task, between
from judgeval.v1.tracer.tracer import Tracer
import time

class TracerUser(User):
    wait_time = between(0.001, 0.01)  # High frequency

    def on_start(self):
        self.tracer = Tracer(
            project_name="load-test",
            enable_monitoring=True,
        )

    @task(10)
    def trace_fast_function(self):
        @self.tracer.observe(span_type="function")
        def fast_function():
            return "result"
        fast_function()

    @task(5)
    def trace_with_evaluation(self):
        @self.tracer.observe(span_type="function")
        def evaluated_function():
            self.tracer.async_evaluate(...)
            return "result"
        evaluated_function()

    @task(3)
    def trace_with_tags(self):
        with self.tracer.span("tagged-span"):
            self.tracer.tag(["test", "load"])

    @task(1)
    def nested_spans(self):
        @self.tracer.observe(span_type="function")
        def outer():
            @self.tracer.observe(span_type="function")
            def inner():
                return "deep"
            return inner()
        outer()
```

**Run Command:**
```bash
locust -f src/tests/reliability/load_tests/locustfile.py \
       --users 1000 \
       --spawn-rate 100 \
       --run-time 10m \
       --headless
```

---

### 2.3 Chaos Engineering Tests

#### Test Suite 6: Chaos Scenarios
**File:** `src/tests/reliability/test_chaos.py`

Tests to add:
1. **test_random_api_timeouts**
   - 20% of API calls timeout randomly
   - Verify SDK maintains stability
   - Verify customer code unaffected

2. **test_intermittent_network_partitions**
   - Simulate network dropping 50% of packets
   - Verify traces eventually reach backend
   - Verify no SDK crashes

3. **test_memory_pressure**
   - Use cgroups to limit SDK memory to 100MB
   - Generate high trace volume
   - Verify graceful degradation (drop spans, not crash)

4. **test_cpu_throttling**
   - Throttle background threads to 10% CPU
   - Verify customer code unaffected
   - Verify export continues (slowly)

5. **test_concurrent_tracer_shutdown**
   - Shutdown tracer while spans are in-flight
   - Verify no deadlocks or crashes
   - Verify graceful drain or timeout

---

## Part 3: Monitoring & Dashboards

### 3.1 Metrics to Expose

#### Prometheus Metrics Export
**File:** `src/judgeval/v1/tracer/metrics.py`

```python
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

class TracerMetricsExporter:
    """Exposes SDK metrics in Prometheus format."""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Counters
        self.spans_created = Counter(
            'judgeval_spans_created_total',
            'Total spans created',
            registry=self.registry
        )
        self.spans_exported = Counter(
            'judgeval_spans_exported_total',
            'Total spans successfully exported',
            registry=self.registry
        )
        self.spans_dropped = Counter(
            'judgeval_spans_dropped_total',
            'Total spans dropped due to queue full',
            ['reason'],  # reason: queue_full, export_error, serialization_error
            registry=self.registry
        )
        self.api_errors = Counter(
            'judgeval_api_errors_total',
            'Total API errors',
            ['endpoint', 'status_code'],
            registry=self.registry
        )
        self.evaluations_enqueued = Counter(
            'judgeval_evaluations_enqueued_total',
            'Total evaluations enqueued',
            registry=self.registry
        )
        self.evaluations_dropped = Counter(
            'judgeval_evaluations_dropped_total',
            'Total evaluations dropped',
            registry=self.registry
        )
        self.tags_buffered = Counter(
            'judgeval_tags_buffered_total',
            'Total tags buffered',
            registry=self.registry
        )
        self.tags_flushed = Counter(
            'judgeval_tags_flushed_total',
            'Total tags flushed to API',
            registry=self.registry
        )

        # Gauges
        self.queue_size = Gauge(
            'judgeval_queue_size',
            'Current span queue size',
            registry=self.registry
        )
        self.queue_utilization = Gauge(
            'judgeval_queue_utilization_percent',
            'Queue utilization percentage',
            registry=self.registry
        )
        self.metadata_size = Gauge(
            'judgeval_metadata_entries',
            'Number of span metadata entries in memory',
            registry=self.registry
        )
        self.eval_queue_size = Gauge(
            'judgeval_eval_queue_size',
            'Evaluation queue size',
            registry=self.registry
        )
        self.tag_buffer_size = Gauge(
            'judgeval_tag_buffer_size',
            'Tag buffer size',
            registry=self.registry
        )

        # Histograms
        self.export_duration = Histogram(
            'judgeval_export_duration_seconds',
            'Time to export batch of spans',
            registry=self.registry
        )
        self.observe_overhead = Histogram(
            'judgeval_observe_overhead_seconds',
            'Overhead added by @observe decorator',
            registry=self.registry
        )
        self.api_request_duration = Histogram(
            'judgeval_api_request_duration_seconds',
            'API request duration',
            ['endpoint'],
            registry=self.registry
        )

    def export_to_prometheus(self, port: int = 9090):
        """Start Prometheus HTTP server."""
        from prometheus_client import start_http_server
        start_http_server(port, registry=self.registry)
```

**Integration with Tracer:**
```python
class BaseTracer:
    def __init__(self, ...):
        self.metrics_exporter = TracerMetricsExporter()
        if enable_metrics_export:
            self.metrics_exporter.export_to_prometheus(port=9090)
```

---

### 3.2 Grafana Dashboard Configuration

#### Dashboard JSON Template
**File:** `monitoring/grafana/judgeval_sdk_dashboard.json`

**Panels to include:**

1. **Span Throughput Panel**
   - Metric: `rate(judgeval_spans_created_total[1m])`
   - Chart: Time series
   - Alerts: > 5000 spans/sec (yellow), > 10k spans/sec (red)

2. **Span Drop Rate Panel**
   - Metric: `rate(judgeval_spans_dropped_total[5m]) / rate(judgeval_spans_created_total[5m])`
   - Chart: Time series
   - Alerts: > 1% drop rate (yellow), > 5% (red)

3. **Queue Utilization Panel**
   - Metric: `judgeval_queue_utilization_percent`
   - Chart: Gauge
   - Alerts: > 80% (yellow), > 95% (red)

4. **Export Latency Panel**
   - Metric: `histogram_quantile(0.99, judgeval_export_duration_seconds)`
   - Chart: Time series (p50, p95, p99)
   - Alerts: p99 > 10 seconds

5. **API Error Rate Panel**
   - Metric: `rate(judgeval_api_errors_total[5m])`
   - Chart: Time series by endpoint
   - Alerts: > 10 errors/min

6. **Memory Pressure Panel**
   - Metric: `judgeval_metadata_entries`
   - Chart: Time series
   - Alerts: > 8000 entries (yellow), > 9500 (red)

7. **Evaluation Queue Panel**
   - Metric: `judgeval_eval_queue_size`
   - Chart: Gauge
   - Alerts: > 800 (yellow), > 950 (red)

8. **Observer Overhead Panel**
   - Metric: `histogram_quantile(0.95, judgeval_observe_overhead_seconds)`
   - Chart: Time series
   - Alerts: p95 > 1ms

**Dashboard Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Judgeval SDK Health Dashboard                          │
├──────────────────────┬──────────────────────────────────┤
│ Span Throughput      │  Span Drop Rate                  │
│ (spans/sec)          │  (%)                             │
├──────────────────────┼──────────────────────────────────┤
│ Queue Utilization    │  Export Latency (p99)            │
│ (%)                  │  (seconds)                       │
├──────────────────────┼──────────────────────────────────┤
│ API Error Rate       │  Memory Pressure                 │
│ (errors/min)         │  (metadata entries)              │
├──────────────────────┴──────────────────────────────────┤
│ Evaluation Queue Depth                                  │
│ (queued evaluations)                                    │
├─────────────────────────────────────────────────────────┤
│ Observer Overhead (p95 latency)                         │
│ (milliseconds)                                          │
└─────────────────────────────────────────────────────────┘
```

**Grafana Dashboard Import:**
```bash
# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/judgeval_sdk_dashboard.json
```

---

### 3.3 Alerting Rules

#### Prometheus Alert Rules
**File:** `monitoring/prometheus/alerts.yml`

```yaml
groups:
  - name: judgeval_sdk_alerts
    interval: 30s
    rules:
      # Critical Alerts
      - alert: JudgevalHighDropRate
        expr: rate(judgeval_spans_dropped_total[5m]) / rate(judgeval_spans_created_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High span drop rate detected"
          description: "{{ $value | humanizePercentage }} of spans are being dropped (threshold: 5%)"

      - alert: JudgevalQueueSaturated
        expr: judgeval_queue_utilization_percent > 95
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Span queue nearly full"
          description: "Queue utilization at {{ $value }}% (threshold: 95%)"

      - alert: JudgevalExportStalled
        expr: time() - judgeval_last_export_timestamp > 60
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Span export has stalled"
          description: "No spans exported in the last minute"

      # Warning Alerts
      - alert: JudgevalHighAPIErrorRate
        expr: rate(judgeval_api_errors_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API error rate"
          description: "{{ $value }} API errors per minute (threshold: 10)"

      - alert: JudgevalHighObserveOverhead
        expr: histogram_quantile(0.95, judgeval_observe_overhead_seconds) > 0.001
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High @observe overhead"
          description: "p95 overhead is {{ $value }}s (threshold: 1ms)"

      - alert: JudgevalMemoryPressure
        expr: judgeval_metadata_entries > 9000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High span metadata memory usage"
          description: "{{ $value }} metadata entries in memory (threshold: 9k)"

      - alert: JudgevalEvaluationQueueFull
        expr: judgeval_eval_queue_size > 900
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Evaluation queue nearly full"
          description: "{{ $value }} evaluations queued (threshold: 900/1000)"
```

---

### 3.4 Setting Up the Monitoring Stack

#### Docker Compose for Local Testing
**File:** `monitoring/docker-compose.yml`

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - ./grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
      - ./grafana/judgeval_sdk_dashboard.json:/etc/grafana/provisioning/dashboards/judgeval.json
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/config.yml:/etc/alertmanager/config.yml
      - alertmanager_data:/alertmanager

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
```

**Prometheus Configuration:**
**File:** `monitoring/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/alerts.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  - job_name: 'judgeval_sdk'
    static_configs:
      - targets:
          - host.docker.internal:9090  # SDK metrics endpoint
```

**Grafana Datasource:**
**File:** `monitoring/grafana/datasources.yml`

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

**Start Monitoring Stack:**
```bash
cd monitoring
docker-compose up -d

# Access Grafana: http://localhost:3000 (admin/admin)
# Access Prometheus: http://localhost:9091
```

---

### 3.5 Integration Test with Monitoring

#### Test with Live Metrics
**File:** `src/tests/reliability/test_with_monitoring.py`

```python
import pytest
import requests
import time
from judgeval.v1.tracer.tracer import Tracer

@pytest.mark.monitoring
class TestSDKWithMonitoring:
    """Tests that verify metrics are correctly exported."""

    def test_metrics_endpoint_available(self):
        """Verify Prometheus metrics endpoint is accessible."""
        response = requests.get("http://localhost:9090/metrics")
        assert response.status_code == 200
        assert "judgeval_spans_created_total" in response.text

    def test_span_creation_increments_counter(self, tracer: Tracer):
        """Verify span creation increments the counter."""
        # Get baseline
        response = requests.get("http://localhost:9090/metrics")
        baseline = self._extract_metric(response.text, "judgeval_spans_created_total")

        # Create spans
        for i in range(100):
            with tracer.span(f"test-{i}"):
                pass

        time.sleep(2)  # Allow metrics to update

        # Verify increment
        response = requests.get("http://localhost:9090/metrics")
        current = self._extract_metric(response.text, "judgeval_spans_created_total")

        assert current >= baseline + 100

    def test_queue_utilization_tracked(self, tracer: Tracer):
        """Verify queue utilization is tracked."""
        # Saturate queue
        for i in range(5000):
            with tracer.span(f"saturate-{i}"):
                pass

        time.sleep(1)

        response = requests.get("http://localhost:9090/metrics")
        utilization = self._extract_metric(response.text, "judgeval_queue_utilization_percent")

        assert utilization > 0  # Should have some utilization

    @staticmethod
    def _extract_metric(metrics_text: str, metric_name: str) -> float:
        """Extract metric value from Prometheus text format."""
        for line in metrics_text.split('\n'):
            if line.startswith(metric_name) and not line.startswith('#'):
                return float(line.split()[-1])
        return 0.0
```

---

## Part 4: Implementation Roadmap

### Phase 1: Critical Fixes (Week 1-2)
**Priority: Must have before production use**

1. **Implement Background Evaluation Queue** (Problem 1)
   - [ ] Add thread-safe queue for evaluations
   - [ ] Background worker thread
   - [ ] Unit tests for queue behavior
   - [ ] Integration tests with API mocking

2. **Implement Tag Buffering** (Problem 2)
   - [ ] Add tag buffer with periodic flush
   - [ ] Backend endpoint `/traces/tags/add_batch`
   - [ ] Unit tests for batching logic
   - [ ] Integration tests

3. **Add Span Metadata Eviction** (Problem 3)
   - [ ] Implement TTL + LRU eviction
   - [ ] Background cleanup thread
   - [ ] Unit tests for eviction policy
   - [ ] Memory leak test validation

4. **Add SDK Metrics** (Problem 4)
   - [ ] Implement `TracerMetrics` class
   - [ ] Hook metrics into span processor
   - [ ] Add `get_health_metrics()` API
   - [ ] Unit tests for metric tracking

### Phase 2: Monitoring Infrastructure (Week 3)
**Priority: High (needed for production observability)**

1. **Prometheus Integration**
   - [ ] Implement `TracerMetricsExporter`
   - [ ] Add Prometheus client dependency
   - [ ] Expose `/metrics` endpoint
   - [ ] Write integration tests

2. **Grafana Dashboard**
   - [ ] Create dashboard JSON
   - [ ] Test with sample data
   - [ ] Document setup process
   - [ ] Add screenshots to README

3. **Alerting Rules**
   - [ ] Write Prometheus alert rules
   - [ ] Configure Alertmanager
   - [ ] Test alert firing
   - [ ] Document runbook

### Phase 3: Enhanced Testing (Week 4-5)
**Priority: Medium (validation of fixes)**

1. **API Failure Resilience Tests**
   - [ ] Write 5 tests in `test_api_failures.py`
   - [ ] Mock API with various failure modes
   - [ ] Validate non-blocking behavior

2. **Sustained Load Tests**
   - [ ] Write 5 tests in `test_sustained_load.py`
   - [ ] Setup Locust load testing
   - [ ] Run 60-second sustained load test
   - [ ] Validate memory stability

3. **Crash Resistance Tests**
   - [ ] Write 4 tests in `test_crash_resistance.py`
   - [ ] Inject SDK failures
   - [ ] Validate exception isolation

4. **Memory Leak Validation**
   - [ ] Expand `test_memory_leaks.py`
   - [ ] Add hung span test
   - [ ] Add metadata eviction test
   - [ ] Run 100k iteration tests

### Phase 4: Advanced Features (Week 6+)
**Priority: Low (nice to have)**

1. **Retry Logic**
   - [ ] Add tenacity dependency
   - [ ] Implement exponential backoff
   - [ ] Configurable retry policy
   - [ ] Tests for retry behavior

2. **Configurable Queue Sizes**
   - [ ] Add parameters to `TracerFactory.create()`
   - [ ] Document configuration options
   - [ ] Add configuration validation

3. **Chaos Engineering Tests**
   - [ ] Setup chaos testing framework
   - [ ] Write 5 chaos scenarios
   - [ ] Run in CI/CD pipeline

4. **24-Hour Soak Test**
   - [ ] Setup long-running test harness
   - [ ] Run in dedicated environment
   - [ ] Automated memory/performance reporting

---

## Part 5: Success Metrics

### Key Performance Indicators (KPIs)

1. **Zero Customer Impact**
   - `@observe` overhead: < 1ms (p95)
   - No exceptions raised to customer code
   - No blocking on API calls

2. **High Reliability**
   - Span drop rate: < 0.1% under normal load
   - Span drop rate: < 5% under 10x load spike
   - Memory usage: stable over 24 hours

3. **Observable Health**
   - All critical metrics exposed via Prometheus
   - Alerting on degraded conditions
   - Dashboard shows real-time SDK health

4. **Test Coverage**
   - Unit test coverage: > 90%
   - Reliability test coverage: 100% of critical paths
   - Load tests pass at 10k spans/sec

---

## Part 6: Documentation Updates Needed

1. **Reliability Documentation** (`docs/reliability.md`)
   - Document SDK guarantees
   - Explain failure modes
   - Configuration best practices

2. **Monitoring Guide** (`docs/monitoring.md`)
   - How to enable Prometheus metrics
   - Grafana dashboard setup
   - Interpreting metrics

3. **Performance Tuning** (`docs/performance.md`)
   - Queue size configuration
   - When to adjust batch sizes
   - High-throughput recommendations

4. **Troubleshooting** (`docs/troubleshooting.md`)
   - High drop rate diagnosis
   - Memory pressure investigation
   - API timeout debugging

---

## Appendix A: Risk Assessment Summary

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Blocking API calls | **Critical** | High | Background queues (Phase 1) |
| Memory leak | **High** | Medium | TTL eviction (Phase 1) |
| Silent trace loss | **High** | High | Metrics + alerts (Phase 2) |
| Queue saturation | **Medium** | Medium | Configurable sizes (Phase 4) |
| API timeout cascades | **Medium** | Low | Retry logic (Phase 4) |

---

## Appendix B: Testing Checklist

### Before Production Release
- [ ] All Phase 1 critical fixes implemented
- [ ] Phase 2 monitoring deployed
- [ ] All new tests passing in CI/CD
- [ ] Load tests validated at 10k spans/sec
- [ ] 24-hour soak test completed (optional)
- [ ] Documentation updated
- [ ] Grafana dashboard tested with real data
- [ ] Alert rules validated
- [ ] Customer-facing SDK guarantees documented

### Ongoing Monitoring (Post-Release)
- [ ] Weekly review of Grafana dashboards
- [ ] Monthly load testing
- [ ] Quarterly chaos testing
- [ ] Customer feedback on SDK impact

---

## Contact & Questions

For questions about this analysis:
- **SDK Team Lead:** [Your name]
- **Reliability Engineer:** [Your name]
- **Documentation:** `RELIABILITY_ANALYSIS.md` (this file)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-27
**Next Review:** 2026-02-27
