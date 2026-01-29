# Prometheus & Grafana Integration Plan

This document outlines the plan to export reliability test metrics to Prometheus and visualize them in Grafana.

## Architecture Overview

```
┌─────────────────────┐
│  Reliability Tests  │
│   (pytest runs)     │
└──────────┬──────────┘
           │ Metrics
           ↓
┌─────────────────────┐
│  Pytest Plugin      │
│  (custom plugin)    │
└──────────┬──────────┘
           │ HTTP POST
           ↓
┌─────────────────────┐
│ Prometheus Pushgateway │
│  (temporary storage)   │
└──────────┬──────────┘
           │ Scrape
           ↓
┌─────────────────────┐
│    Prometheus       │
│  (time-series DB)   │
└──────────┬──────────┘
           │ Query
           ↓
┌─────────────────────┐
│     Grafana         │
│  (visualization)    │
└─────────────────────┘
```

## Phase 1: Infrastructure Setup

### 1.1 Deploy Prometheus

**Option A: Docker Compose (Development/Testing)**

```yaml
# docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=90d'
    restart: unless-stopped

  pushgateway:
    image: prom/pushgateway:latest
    container_name: pushgateway
    ports:
      - "9091:9091"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

**Option B: Kubernetes (Production)**

```yaml
# k8s/prometheus-deployment.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: storage
          mountPath: /prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: storage
        persistentVolumeClaim:
          claimName: prometheus-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  type: LoadBalancer
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
```

### 1.2 Configure Prometheus

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: 'staging'
    cluster: 'reliability-tests'

scrape_configs:
  # Scrape Pushgateway
  - job_name: 'pushgateway'
    honor_labels: true
    static_configs:
      - targets: ['pushgateway:9091']

  # Scrape Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

# Alerting rules
rule_files:
  - '/etc/prometheus/alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### 1.3 Setup Grafana Data Source

```yaml
# grafana/provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

## Phase 2: Pytest Plugin Development

### 2.1 Create Custom Pytest Plugin

```python
# src/tests/reliability/plugins/pytest_prometheus.py
"""
Pytest plugin to export test metrics to Prometheus Pushgateway.
"""

import time
import pytest
import psutil
import tracemalloc
from typing import Dict, Any
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    Histogram,
    push_to_gateway,
)


class PrometheusReporter:
    """Collects and exports pytest metrics to Prometheus."""

    def __init__(self, pushgateway_url: str, job_name: str = "reliability_tests"):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.registry = CollectorRegistry()

        # Define metrics
        self.test_duration = Histogram(
            'reliability_test_duration_seconds',
            'Test execution duration',
            ['test_name', 'test_class', 'test_file', 'outcome'],
            registry=self.registry,
        )

        self.test_memory_usage = Gauge(
            'reliability_test_memory_mb',
            'Memory usage during test',
            ['test_name', 'test_class', 'measurement_type'],
            registry=self.registry,
        )

        self.test_outcome = Counter(
            'reliability_test_outcome_total',
            'Test outcome counts',
            ['test_name', 'outcome'],
            registry=self.registry,
        )

        self.test_assertions = Counter(
            'reliability_test_assertions_total',
            'Custom assertions tracked',
            ['test_name', 'assertion_type'],
            registry=self.registry,
        )

        # Performance metrics
        self.latency_measurement = Gauge(
            'reliability_latency_ms',
            'Latency measurements from tests',
            ['test_name', 'operation', 'percentile'],
            registry=self.registry,
        )

        self.throughput_measurement = Gauge(
            'reliability_throughput_ops_per_sec',
            'Throughput measurements from tests',
            ['test_name', 'operation'],
            registry=self.registry,
        )

        self.span_count = Counter(
            'reliability_spans_generated_total',
            'Total spans generated during tests',
            ['test_name'],
            registry=self.registry,
        )

        # System metrics
        self.cpu_usage = Gauge(
            'reliability_test_cpu_percent',
            'CPU usage during test',
            ['test_name'],
            registry=self.registry,
        )

        self.memory_growth = Gauge(
            'reliability_memory_growth_mb',
            'Memory growth during test',
            ['test_name'],
            registry=self.registry,
        )

        # Test metadata
        self.test_run_timestamp = Gauge(
            'reliability_test_run_timestamp',
            'Timestamp of test run',
            ['git_branch', 'git_commit'],
            registry=self.registry,
        )

        # Active test context
        self.current_test: Dict[str, Any] = {}

    def pytest_runtest_setup(self, item):
        """Called before each test starts."""
        self.current_test = {
            'name': item.name,
            'class': item.cls.__name__ if item.cls else 'module',
            'file': item.fspath.basename,
            'start_time': time.time(),
            'process': psutil.Process(),
        }

        # Start memory tracking
        tracemalloc.start()
        self.current_test['baseline_memory'] = tracemalloc.take_snapshot()

    def pytest_runtest_teardown(self, item, nextitem):
        """Called after each test completes."""
        if not self.current_test:
            return

        duration = time.time() - self.current_test['start_time']
        test_name = self.current_test['name']
        test_class = self.current_test['class']
        test_file = self.current_test['file']

        # Record duration
        outcome = self.current_test.get('outcome', 'unknown')
        self.test_duration.labels(
            test_name=test_name,
            test_class=test_class,
            test_file=test_file,
            outcome=outcome,
        ).observe(duration)

        # Record outcome
        self.test_outcome.labels(
            test_name=test_name,
            outcome=outcome,
        ).inc()

        # Record memory
        if tracemalloc.is_tracing():
            final_snapshot = tracemalloc.take_snapshot()
            baseline_snapshot = self.current_test['baseline_memory']

            baseline_size = sum(stat.size for stat in baseline_snapshot.statistics("filename"))
            final_size = sum(stat.size for stat in final_snapshot.statistics("filename"))

            memory_growth_mb = (final_size - baseline_size) / 1024 / 1024

            self.memory_growth.labels(test_name=test_name).set(memory_growth_mb)
            tracemalloc.stop()

        # Record CPU usage
        try:
            cpu_percent = self.current_test['process'].cpu_percent()
            self.cpu_usage.labels(test_name=test_name).set(cpu_percent)
        except:
            pass

        # Push metrics to Pushgateway
        try:
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=self.registry,
                grouping_key={'test': test_name},
            )
        except Exception as e:
            print(f"Failed to push metrics: {e}")

        self.current_test = {}

    def pytest_runtest_makereport(self, item, call):
        """Called when test report is created."""
        if call.when == 'call':
            self.current_test['outcome'] = call.outcome

    def record_custom_metric(self, metric_name: str, value: float, labels: Dict[str, str]):
        """Allow tests to record custom metrics."""
        if metric_name == 'latency':
            self.latency_measurement.labels(**labels).set(value)
        elif metric_name == 'throughput':
            self.throughput_measurement.labels(**labels).set(value)
        elif metric_name == 'span_count':
            self.span_count.labels(**labels).inc(value)


# Global reporter instance
_reporter = None


def pytest_configure(config):
    """Initialize plugin."""
    global _reporter

    pushgateway_url = config.getoption('--pushgateway-url', default=None)
    if pushgateway_url:
        _reporter = PrometheusReporter(pushgateway_url)
        config.pluginmanager.register(_reporter, 'prometheus_reporter')


def pytest_addoption(parser):
    """Add command-line options."""
    parser.addoption(
        '--pushgateway-url',
        action='store',
        default=None,
        help='Prometheus Pushgateway URL (e.g., http://localhost:9091)',
    )


@pytest.fixture
def metrics_reporter():
    """Fixture to allow tests to record custom metrics."""
    return _reporter
```

### 2.2 Update conftest.py

```python
# src/tests/reliability/conftest.py (additions)

@pytest.fixture
def record_latency(metrics_reporter):
    """Helper to record latency measurements."""
    def _record(operation: str, latency_ms: float, percentile: str = 'p50'):
        if metrics_reporter:
            test_name = pytest.current_test_name()  # Need to implement
            metrics_reporter.latency_measurement.labels(
                test_name=test_name,
                operation=operation,
                percentile=percentile,
            ).set(latency_ms)
    return _record


@pytest.fixture
def record_throughput(metrics_reporter):
    """Helper to record throughput measurements."""
    def _record(operation: str, ops_per_sec: float):
        if metrics_reporter:
            test_name = pytest.current_test_name()
            metrics_reporter.throughput_measurement.labels(
                test_name=test_name,
                operation=operation,
            ).set(ops_per_sec)
    return _record
```

### 2.3 Update pytest.ini

```ini
# pytest.ini (additions)

[pytest]
# ... existing config ...

# Prometheus plugin
addopts =
    --pushgateway-url=http://localhost:9091
    # or via env var: --pushgateway-url=${PROMETHEUS_PUSHGATEWAY_URL}
```

## Phase 3: Instrument Tests

### 3.1 Add Metric Recording to Tests

```python
# Example: Update test_latency.py
def test_observe_adds_minimal_overhead(self, tracer: Tracer, record_latency):
    """Single traced call should add < 1ms overhead."""
    ITERATIONS = 100
    MAX_OVERHEAD_MS = 1.0

    # ... existing test code ...

    baseline_median = statistics.median(baseline_times)
    traced_median = statistics.median(traced_times)
    overhead = traced_median - baseline_median

    # Record metrics
    record_latency('observe_decorator', baseline_median, 'p50')
    record_latency('observe_decorator_traced', traced_median, 'p50')
    record_latency('observe_decorator_overhead', overhead, 'p50')

    # Calculate percentiles
    p95_overhead = statistics.quantiles(
        [t - b for t, b in zip(traced_times, baseline_times)], n=20
    )[18]
    record_latency('observe_decorator_overhead', p95_overhead, 'p95')

    assert overhead < MAX_OVERHEAD_MS
```

### 3.2 Create Metric Helper Functions

```python
# src/tests/reliability/helpers/metrics.py
"""Helper functions for recording metrics in tests."""

import statistics
from typing import List


def record_latency_distribution(
    latencies: List[float],
    operation: str,
    record_func,
):
    """Record latency percentiles."""
    if not latencies:
        return

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    record_func(operation, latencies_sorted[n // 2], 'p50')
    record_func(operation, latencies_sorted[int(n * 0.95)], 'p95')
    record_func(operation, latencies_sorted[int(n * 0.99)], 'p99')
    record_func(operation, statistics.mean(latencies), 'mean')


def record_memory_stats(
    baseline_mb: float,
    final_mb: float,
    operation: str,
    metrics_reporter,
):
    """Record memory statistics."""
    test_name = metrics_reporter.current_test['name']

    metrics_reporter.test_memory_usage.labels(
        test_name=test_name,
        test_class=metrics_reporter.current_test['class'],
        measurement_type='baseline',
    ).set(baseline_mb)

    metrics_reporter.test_memory_usage.labels(
        test_name=test_name,
        test_class=metrics_reporter.current_test['class'],
        measurement_type='final',
    ).set(final_mb)

    growth_mb = final_mb - baseline_mb
    metrics_reporter.memory_growth.labels(test_name=test_name).set(growth_mb)
```

## Phase 4: Grafana Dashboards

### 4.1 Main Reliability Dashboard

```json
// grafana/provisioning/dashboards/reliability-overview.json
{
  "dashboard": {
    "title": "SDK Reliability Overview",
    "panels": [
      {
        "title": "Test Success Rate (24h)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(reliability_test_outcome_total{outcome=\"passed\"}[24h])) / sum(rate(reliability_test_outcome_total[24h])) * 100"
          }
        ]
      },
      {
        "title": "Test Duration Trends",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(reliability_test_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Memory Growth by Test",
        "type": "graph",
        "targets": [
          {
            "expr": "reliability_memory_growth_mb"
          }
        ]
      },
      {
        "title": "Latency P95 (Observe Decorator)",
        "type": "graph",
        "targets": [
          {
            "expr": "reliability_latency_ms{operation=\"observe_decorator\", percentile=\"p95\"}"
          }
        ]
      }
    ]
  }
}
```

### 4.2 Performance Dashboard

```json
// grafana/provisioning/dashboards/performance.json
{
  "dashboard": {
    "title": "SDK Performance Metrics",
    "panels": [
      {
        "title": "Latency Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(reliability_test_duration_seconds_bucket[5m])"
          }
        ]
      },
      {
        "title": "Throughput Over Time",
        "type": "graph",
        "targets": [
          {
            "expr": "reliability_throughput_ops_per_sec"
          }
        ]
      },
      {
        "title": "CPU Usage During Tests",
        "type": "graph",
        "targets": [
          {
            "expr": "reliability_test_cpu_percent"
          }
        ]
      }
    ]
  }
}
```

### 4.3 E2E Test Dashboard

```json
// grafana/provisioning/dashboards/e2e-tests.json
{
  "dashboard": {
    "title": "E2E Test Results",
    "panels": [
      {
        "title": "Trace Delivery Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(reliability_test_outcome_total{test_name=~\"test_e2e.*\", outcome=\"passed\"}) / sum(reliability_test_outcome_total{test_name=~\"test_e2e.*\"})"
          }
        ]
      },
      {
        "title": "Spans Delivered vs Expected",
        "type": "graph",
        "targets": [
          {
            "expr": "reliability_spans_generated_total"
          },
          {
            "expr": "reliability_spans_delivered_total"
          }
        ]
      }
    ]
  }
}
```

## Phase 5: Alerting

### 5.1 Define Alert Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: reliability_tests
    interval: 30s
    rules:
      # Test failure alerts
      - alert: ReliabilityTestFailureRate
        expr: |
          sum(rate(reliability_test_outcome_total{outcome="failed"}[1h]))
          /
          sum(rate(reliability_test_outcome_total[1h]))
          > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High test failure rate"
          description: "{{ $value | humanizePercentage }} of reliability tests are failing"

      # Performance degradation
      - alert: LatencyDegradation
        expr: |
          reliability_latency_ms{percentile="p95", operation="observe_decorator"} > 2.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Observe decorator latency increased"
          description: "P95 latency is {{ $value }}ms (threshold: 2ms)"

      # Memory leak detection
      - alert: MemoryLeakDetected
        expr: |
          reliability_memory_growth_mb > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Potential memory leak in {{ $labels.test_name }}"
          description: "Memory grew by {{ $value }}MB"

      # E2E test failures
      - alert: E2ETestsFailing
        expr: |
          sum(rate(reliability_test_outcome_total{test_name=~"test_e2e.*", outcome="failed"}[30m])) > 0
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "E2E tests are failing"
          description: "One or more E2E tests have been failing for 15+ minutes"

      # Throughput degradation
      - alert: ThroughputDegradation
        expr: |
          reliability_throughput_ops_per_sec < 8000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Throughput below threshold"
          description: "Current throughput: {{ $value }} ops/sec (threshold: 8000)"
```

### 5.2 Configure AlertManager

```yaml
# alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'slack-notifications'

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    - match:
        severity: warning
      receiver: 'slack-notifications'

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - channel: '#sdk-alerts'
        title: 'Reliability Test Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

## Phase 6: CI/CD Integration

### 6.1 Update GitHub Actions Workflow

```yaml
# .github/workflows/reliability-tests.yml (additions)
jobs:
  reliability-tests-with-metrics:
    runs-on: ubuntu-latest

    services:
      pushgateway:
        image: prom/pushgateway:latest
        ports:
          - 9091:9091

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install prometheus-client psutil

      - name: Run reliability tests with metrics
        env:
          PROMETHEUS_PUSHGATEWAY_URL: http://localhost:9091
        run: |
          pytest src/tests/reliability/ \
            -v \
            -m "reliability and not e2e and not slow" \
            --pushgateway-url=$PROMETHEUS_PUSHGATEWAY_URL

      - name: Push final metrics
        if: always()
        run: |
          # Push CI metadata
          python scripts/push_ci_metadata.py
```

### 6.2 Create CI Metadata Script

```python
# scripts/push_ci_metadata.py
"""Push CI metadata to Prometheus."""

import os
import subprocess
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

registry = CollectorRegistry()

# Git metadata
git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

# CI metadata
ci_run_timestamp = Gauge(
    'reliability_ci_run_timestamp',
    'Timestamp of CI run',
    ['git_branch', 'git_commit', 'workflow'],
    registry=registry,
)

ci_run_timestamp.labels(
    git_branch=git_branch,
    git_commit=git_commit,
    workflow=os.getenv('GITHUB_WORKFLOW', 'local'),
).set_to_current_time()

# Push to gateway
pushgateway_url = os.getenv('PROMETHEUS_PUSHGATEWAY_URL', 'http://localhost:9091')
push_to_gateway(pushgateway_url, job='ci_metadata', registry=registry)
```

## Phase 7: Long-term Storage & Analysis

### 7.1 Configure Prometheus Long-term Storage

```yaml
# prometheus/prometheus.yml (additions)
storage:
  tsdb:
    retention:
      time: 90d  # Keep 90 days of data
      size: 50GB  # Max 50GB storage
```

### 7.2 Setup Remote Write (Optional)

For longer-term storage, configure remote write to a time-series database:

```yaml
# prometheus/prometheus.yml (additions)
remote_write:
  - url: "https://prometheus-prod-10-prod-us-east-0.grafana.net/api/prom/push"
    basic_auth:
      username: "your-username"
      password: "your-api-key"
```

## Implementation Timeline

### Week 1: Infrastructure
- [ ] Deploy Prometheus + Pushgateway (Docker Compose locally)
- [ ] Deploy Grafana
- [ ] Configure datasources

### Week 2: Plugin Development
- [ ] Create pytest-prometheus plugin
- [ ] Add command-line options
- [ ] Test metric collection locally

### Week 3: Test Instrumentation
- [ ] Instrument latency tests
- [ ] Instrument memory tests
- [ ] Instrument concurrency tests
- [ ] Instrument E2E tests

### Week 4: Dashboards
- [ ] Create main reliability dashboard
- [ ] Create performance dashboard
- [ ] Create E2E dashboard
- [ ] Test visualization

### Week 5: Alerting
- [ ] Define alert rules
- [ ] Configure AlertManager
- [ ] Test Slack/PagerDuty integration
- [ ] Document runbooks

### Week 6: CI Integration
- [ ] Update GitHub Actions
- [ ] Add Pushgateway service
- [ ] Test end-to-end flow
- [ ] Document usage

## Success Metrics

After implementation, you should be able to:

1. **View real-time metrics**: See test execution metrics in Grafana within seconds
2. **Track trends**: Identify performance regressions over time (weeks/months)
3. **Get alerted**: Receive Slack/PagerDuty alerts when tests fail or degrade
4. **Debug issues**: Correlate test failures with specific commits/branches
5. **Capacity planning**: Understand resource requirements under load

## Resources

- Prometheus Python Client: https://github.com/prometheus/client_python
- Grafana Dashboards: https://grafana.com/docs/grafana/latest/dashboards/
- Pytest Plugins: https://docs.pytest.org/en/latest/how-to/writing_plugins.html
- Pushgateway: https://github.com/prometheus/pushgateway
