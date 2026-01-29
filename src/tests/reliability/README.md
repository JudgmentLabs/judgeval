# Reliability Tests

This directory contains comprehensive reliability, performance, and stress tests for the Judgeval v1 SDK. These tests ensure the SDK is production-ready, handles edge cases gracefully, and maintains performance under load.

## Overview

The reliability test suite is organized into two categories:

1. **Unit/Integration Tests (Mocked)**: Fast tests that use mocked API clients to verify SDK behavior in isolation
2. **End-to-End Tests**: Tests that run against staging/production to verify actual trace delivery and data integrity

## Test Categories

### 1. Sustained Load Tests (`test_sustained_load.py`)

Tests SDK behavior under sustained high load conditions.

**Tests:**
- `test_100k_spans_over_60_seconds` - Generates 100k spans over 60s (~1600 spans/sec) and monitors memory growth
- `test_burst_traffic_spike` - Simulates sudden traffic spike (10k spans in 1 second)
- `test_queue_saturation_under_load` - Generates spans faster than export capacity
- `test_concurrent_high_frequency_threads` - 100 threads each generating 1000 spans
- `test_long_running_service_24_hours` - 24-hour stability test (skipped by default)

**Marks:** `@pytest.mark.reliability`, `@pytest.mark.slow`

**Run:**
```bash
pytest src/tests/reliability/test_sustained_load.py -v
pytest src/tests/reliability/test_sustained_load.py -v -m "reliability and not slow"
```

### 2. API Failure Resilience (`test_api_failures.py`)

Verifies SDK gracefully handles API failures without blocking customer code.

**Tests:**
- `test_evaluation_enqueue_with_slow_api` - Ensures async_evaluate doesn't block when API is slow (<100ms)
- `test_evaluation_enqueue_with_api_down` - Evaluation failures are isolated from customer code
- `test_tag_with_api_timeout` - Tagging doesn't block on timeout (<50ms)
- `test_trace_export_with_network_errors` - Export retries on transient network errors
- `test_api_rate_limiting` - SDK backs off on 429 rate limit responses

**Run:**
```bash
pytest src/tests/reliability/test_api_failures.py -v
```

### 3. Chaos Engineering (`test_chaos.py`)

Tests robustness to unpredictable failures.

**Tests:**
- `test_random_api_timeouts` - 20% of API calls timeout randomly
- `test_intermittent_network_partitions` - Network drops 50% of export attempts
- `test_memory_pressure` - High trace volume under memory constraints (requires cgroups)
- `test_cpu_throttling` - Customer code remains fast under background pressure
- `test_concurrent_tracer_shutdown` - Shutdown while spans are in-flight

**Marks:** `@pytest.mark.chaos`

**Run:**
```bash
pytest src/tests/reliability/test_chaos.py -v -m chaos
```

### 4. Crash Resistance (`test_crash_resistance.py`)

Ensures SDK failures never crash customer applications.

**Tests:**
- `test_sdk_never_crashes_customer_code` - Various SDK failures don't affect customer code
- `test_oom_during_serialization` - Large payloads (10MB) handled gracefully
- `test_latency_with_background_pressure` - Background export doesn't affect foreground latency (P95 <10ms)

**Run:**
```bash
pytest src/tests/reliability/test_crash_resistance.py -v
```

### 5. Edge Cases (`test_edge_cases.py`)

Tests unusual but valid inputs.

**Test Classes:**
- `TestLargePayloads` - 1MB+ strings, 10k item dicts, 100k item lists, large nested structures
- `TestDeeplyNestedData` - 100-level nested dicts/lists, mixed structures
- `TestSpecialCharacters` - Unicode (Chinese, Arabic, Emoji), control characters, mixed encodings
- `TestCircularReferences` - Circular dict/list references, mutual references
- `TestBoundaryConditions` - Empty inputs, None values, extreme numerics (inf, nan), boolean edge cases
- `TestMultipleTracerInstances` - Multiple tracers with different configs
- `TestUnusualFunctionSignatures` - *args/**kwargs, defaults, lambdas, class methods, static methods

**Run:**
```bash
pytest src/tests/reliability/test_edge_cases.py -v
```

### 6. Failure Isolation (`test_isolation.py`)

Verifies SDK failures don't impact customer code.

**Test Classes:**
- `TestAPIFailureIsolation` - API timeouts, HTTP 500s, network errors don't crash user code
- `TestInitializationFailureIsolation` - Project resolution failures, invalid credentials handled gracefully
- `TestSerializationFailureIsolation` - Unserializable data doesn't crash
- `TestExceptionIsolation` - User exceptions propagate unchanged through @observe
- `TestAsyncEvaluateIsolation` - async_evaluate failures are isolated

**Run:**
```bash
pytest src/tests/reliability/test_isolation.py -v
```

### 7. Latency Tests (`test_latency.py`)

Measures overhead and ensures minimal performance impact.

**Test Classes:**
- `TestObserveOverhead`
  - `test_observe_adds_minimal_overhead` - Single traced call adds <1ms overhead
  - `test_observe_under_concurrent_load` - 1000 concurrent calls maintain low P99 (<100ms)
  - `test_nested_observe_scales_linearly` - 10 levels of nesting, <0.5ms overhead per level
  - `test_high_frequency_tracing` - 10k+ calls/sec throughput
  - `test_monitoring_disabled_adds_zero_overhead` - Disabled monitoring has <0.1ms overhead

- `TestSpanOperationLatency`
  - `test_set_attribute_is_fast` - <0.1ms per call
  - `test_span_context_manager_is_fast` - <0.5ms per span

**Run:**
```bash
pytest src/tests/reliability/test_latency.py -v
```

### 8. Memory Tests (`test_memory.py`)

Ensures no memory leaks and proper resource cleanup.

**Test Classes:**
- `TestMemoryStability`
  - `test_no_memory_leak_over_many_calls` - 100k calls with <50MB growth
  - `test_span_cleanup_on_exception` - Spans cleaned up even on exceptions
  - `test_manual_span_cleanup` - Manual span() context manager cleanup
  - `test_tracer_shutdown_releases_resources` - shutdown() releases resources

- `TestLargePayloadMemory`
  - `test_large_payloads_dont_accumulate` - Large payloads (100KB) don't accumulate in memory
  - `test_attributes_with_large_values` - Large attributes (10KB) handled

- `TestMemoryUnderStress`
  - `test_rapid_span_creation_and_destruction` - 10k rapid spans
  - `test_deeply_nested_spans_memory` - 50 levels deep, 100 iterations

**Run:**
```bash
pytest src/tests/reliability/test_memory.py -v
```

### 9. Concurrency Tests (`test_concurrency.py`)

Verifies thread-safety and concurrent access.

**Test Classes:**
- `TestThreadSafety`
  - `test_concurrent_observe_no_race_conditions` - 100 threads × 100 calls each
  - `test_concurrent_span_creation` - 50 threads × 50 spans
  - `test_context_isolation_between_threads` - Thread-local context isolation
  - `test_thread_pool_executor_compatibility` - 500 tasks with 20 workers

- `TestAsyncConcurrency`
  - `test_async_concurrent_observe` - 100 concurrent async tasks
  - `test_async_span_context_isolation` - Async context isolation
  - `test_mixed_sync_async_tracing` - Mixing sync and async traced functions

- `TestNestedConcurrency`
  - `test_nested_threads_with_tracing` - 10 outer threads × 5 inner threads
  - `test_concurrent_tasks_with_nested_spans` - 20 tasks with 5 levels of nesting

- `TestMultipleTracersConcurrency`
  - `test_multiple_tracers_concurrent_use` - 5 tracers used concurrently

**Run:**
```bash
pytest src/tests/reliability/test_concurrency.py -v
```

### 10. End-to-End Tests (`test_e2e_staging.py`)

Tests against staging environment to verify actual trace delivery.

**Prerequisites:**
```bash
export JUDGMENT_API_KEY="your-api-key"
export JUDGMENT_ORG_ID="your-org-id"
export JUDGMENT_API_URL="https://staging.judgmentlabs.ai/"  # or production
```

**Tests:**
- `test_e2e_sustained_load_1k_spans` - 1k spans over 10s, verify delivery
- `test_e2e_sustained_load_10k_spans` - 10k spans over 30s, verify delivery
- `test_e2e_concurrent_traces` - Multiple concurrent traces
- `test_e2e_data_integrity` - Verify attributes, parent-child relationships, timestamps
- `test_e2e_large_payloads` - Large payloads (1MB) delivered correctly
- `test_e2e_evaluation_delivery` - async_evaluate calls reach platform

**Marks:** `@pytest.mark.e2e`, `@pytest.mark.slow`

**Run:**
```bash
# Skip by default (requires credentials)
pytest src/tests/reliability/ -v -m "not e2e"

# Run E2E tests
pytest src/tests/reliability/test_e2e_staging.py -v -m e2e
```

## Benchmarks

### Microbenchmarks (`benchmarks/bench_overhead.py`)

Uses pytest-benchmark for precise overhead measurements.

**Run:**
```bash
pytest src/tests/reliability/benchmarks/ --benchmark-only
pytest src/tests/reliability/benchmarks/ --benchmark-compare
```

### Load Tests (`load_tests/locustfile.py`)

Uses Locust for distributed load testing.

**Run:**
```bash
locust -f src/tests/reliability/load_tests/locustfile.py

# Then open http://localhost:8089 and configure:
# - Number of users: 1000
# - Spawn rate: 100/sec
# - Duration: 10 minutes
```

## Test Markers

Custom pytest markers for organizing tests:

- `@pytest.mark.reliability` - All reliability tests
- `@pytest.mark.slow` - Long-running tests (>30s)
- `@pytest.mark.chaos` - Chaos engineering tests
- `@pytest.mark.load` - Load tests
- `@pytest.mark.e2e` - End-to-end tests (require credentials)

## Running Tests

### Run All Reliability Tests (Mocked)
```bash
pytest src/tests/reliability/ -v -m "reliability and not e2e and not slow"
```

### Run Fast Tests Only
```bash
pytest src/tests/reliability/ -v -m "reliability and not slow and not e2e"
```

### Run Specific Test Category
```bash
pytest src/tests/reliability/test_latency.py -v
pytest src/tests/reliability/test_memory.py -v
pytest src/tests/reliability/test_concurrency.py -v
```

### Run Chaos Tests
```bash
pytest src/tests/reliability/test_chaos.py -v -m chaos
```

### Run E2E Tests (Requires Credentials)
```bash
export JUDGMENT_API_KEY="..."
export JUDGMENT_ORG_ID="..."
export JUDGMENT_API_URL="https://staging.judgmentlabs.ai/"

pytest src/tests/reliability/test_e2e_staging.py -v -m e2e
```

### Run with Coverage
```bash
pytest src/tests/reliability/ --cov=judgeval.v1 --cov-report=html
```

### Run with Verbose Output
```bash
pytest src/tests/reliability/ -v -s  # -s shows print statements
```

## CI/CD Integration

### GitHub Actions Workflow

The reliability tests run in CI with the following strategy:

1. **Fast Tests (PR checks)**: Run on every PR
   - Excludes: `@pytest.mark.slow`, `@pytest.mark.e2e`
   - Duration: ~2-5 minutes

2. **Full Test Suite (Nightly)**: Run nightly on main branch
   - Includes: All tests except E2E
   - Duration: ~15-30 minutes

3. **E2E Tests (Nightly)**: Run nightly against staging
   - Requires: API credentials stored in GitHub secrets
   - Duration: ~10-20 minutes

4. **Load Tests (Weekly)**: Run weekly using Locust
   - Duration: ~30-60 minutes

### Example GitHub Actions Configuration

```yaml
# .github/workflows/reliability-tests.yml
name: Reliability Tests

on:
  pull_request:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest src/tests/reliability/ -v -m "reliability and not slow and not e2e"

  full-suite:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest src/tests/reliability/ -v -m "not e2e"

  e2e-tests:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest src/tests/reliability/test_e2e_staging.py -v -m e2e
        env:
          JUDGMENT_API_KEY: ${{ secrets.JUDGMENT_API_KEY_STAGING }}
          JUDGMENT_ORG_ID: ${{ secrets.JUDGMENT_ORG_ID_STAGING }}
          JUDGMENT_API_URL: https://staging.judgmentlabs.ai/
```

## Monitoring & Observability

### Prometheus Integration

See `docs/reliability-monitoring.md` for details on:
- Exporting test metrics to Prometheus
- Setting up Grafana dashboards
- Alerting on test failures or performance degradation

### Key Metrics Tracked

- **Latency**: P50, P95, P99 overhead per operation
- **Throughput**: Spans/second, traces/second
- **Memory**: Growth over time, leak detection
- **Reliability**: Test pass rate, error rate
- **Duration**: Test execution time

## Common Issues & Troubleshooting

### Tests Timing Out

If tests timeout, check:
1. Network connectivity to staging/production
2. API rate limits
3. System resources (CPU, memory)

```bash
# Increase timeout for slow tests
pytest src/tests/reliability/ -v --timeout=300
```

### Memory Tests Failing

Memory tests can be sensitive to system state:

```bash
# Run in isolation
pytest src/tests/reliability/test_memory.py -v --forked

# Or with explicit GC
pytest src/tests/reliability/test_memory.py -v --capture=no
```

### E2E Tests Failing

Common issues:
1. Missing/invalid credentials
2. Staging environment down
3. Project limits reached
4. Network issues

```bash
# Verify credentials
python -c "import os; print('API Key:', os.getenv('JUDGMENT_API_KEY')[:10]+'...')"

# Test connectivity
curl -H "Authorization: Bearer $JUDGMENT_API_KEY" $JUDGMENT_API_URL/health
```

## Performance Baselines

Expected performance characteristics (as of v1.0):

| Metric | Target | Measured |
|--------|--------|----------|
| @observe overhead | <1ms | ~0.5ms (P50) |
| span() overhead | <0.5ms | ~0.3ms (P50) |
| set_attribute() | <0.1ms | ~0.05ms |
| Throughput | >10k spans/sec | ~15k spans/sec |
| Memory growth (100k spans) | <50MB | ~30MB |
| Concurrent threads | 100+ | 100 threads ✓ |
| P99 latency (1000 concurrent) | <100ms | ~80ms |

## Contributing

When adding new reliability tests:

1. **Choose the right file**: Add to existing test files when possible
2. **Use appropriate markers**: `@pytest.mark.reliability`, `@pytest.mark.slow`, etc.
3. **Document the test**: Clear docstrings explaining what's being tested
4. **Set clear thresholds**: Use concrete assertions (e.g., `<1ms`, `<50MB`)
5. **Make tests deterministic**: Avoid flaky tests, use proper synchronization
6. **Update this README**: Document new test categories or significant tests

### Example Test Template

```python
@pytest.mark.reliability
class TestNewFeature:
    """Test reliability of new feature."""

    def test_feature_under_load(self, tracer: Tracer):
        """
        Feature should handle 10k operations without memory leak.

        Verifies:
        - No memory growth >20MB
        - All operations complete successfully
        - Latency remains <5ms P99
        """
        MAX_MEMORY_GROWTH_MB = 20
        MAX_P99_LATENCY_MS = 5
        ITERATIONS = 10_000

        # Test implementation
        ...

        assert memory_growth < MAX_MEMORY_GROWTH_MB
        assert p99_latency < MAX_P99_LATENCY_MS
```

## Resources

- **OpenTelemetry Docs**: https://opentelemetry.io/docs/
- **Pytest Docs**: https://docs.pytest.org/
- **Locust Docs**: https://docs.locust.io/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/

## Contact

For questions or issues with reliability tests:
- File an issue: https://github.com/judgmentlabs/judgeval/issues
- Internal Slack: #sdk-reliability
