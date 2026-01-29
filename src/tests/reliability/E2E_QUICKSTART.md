# E2E Tests Quick Start Guide

This guide will help you run end-to-end reliability tests against staging.

## Prerequisites

### 1. Install Dependencies

```bash
cd /path/to/judgeval
pip install -e ".[dev]"
```

### 2. Set Environment Variables

You need valid API credentials for staging:

```bash
export JUDGMENT_API_KEY="your-staging-api-key"
export JUDGMENT_ORG_ID="your-org-id"
export JUDGMENT_API_URL="https://staging.judgmentlabs.ai/"
```

**Important:** Never commit API keys to git. Add them to your shell profile or use a `.env` file (which is gitignored).

### 3. Verify Connectivity

Test that you can reach staging:

```bash
curl -H "Authorization: Bearer $JUDGMENT_API_KEY" \
     "${JUDGMENT_API_URL}/api/v1/health"
```

## Running E2E Tests

### Option 1: Using the Helper Script (Recommended)

```bash
cd src/tests/reliability
./run_e2e_tests.sh
```

This runs all fast E2E tests (excludes `@pytest.mark.slow`).

**Run all tests (including slow):**
```bash
./run_e2e_tests.sh "e2e"
```

**Run specific test:**
```bash
./run_e2e_tests.sh "test_e2e_100_spans_baseline"
```

### Option 2: Using pytest Directly

**Fast tests only:**
```bash
pytest src/tests/reliability/test_e2e_staging.py -v -m "e2e and not slow"
```

**All tests:**
```bash
pytest src/tests/reliability/test_e2e_staging.py -v -m "e2e"
```

**Specific test class:**
```bash
pytest src/tests/reliability/test_e2e_staging.py::TestE2ESustainedLoadSmall -v
```

**Single test:**
```bash
pytest src/tests/reliability/test_e2e_staging.py::TestE2ESustainedLoadSmall::test_e2e_100_spans_baseline -v
```

## Test Structure

### Small Scale Tests (Fast)
These run by default and complete in 1-5 minutes:

- `test_e2e_100_spans_baseline` - 100 spans, verify basic delivery
- `test_e2e_1k_spans_over_10_seconds` - 1k spans paced over 10s
- `test_e2e_attributes_preserved` - Verify data integrity

### Medium Scale Tests (Slow)
These are marked `@pytest.mark.slow` and take 5-15 minutes:

- `test_e2e_10k_spans_over_30_seconds` - 10k spans paced over 30s
- `test_e2e_burst_then_sustained` - 5k burst + 5k sustained
- `test_e2e_concurrent_10_threads` - 10 threads × 500 spans

## Understanding Test Output

### Normal Output

```
[E2E] Created project: e2e-reliability-1234567890-abc123
[E2E] Resolved project_id: proj_xyz789
[E2E] Starting baseline test with 100 spans
[E2E] Test marker: e2e-test-def456
[E2E] Generated 100 spans, forcing flush...
[E2E] Waiting 5s for platform processing...
[E2E] Found 98 matching traces out of 102 total
PASSED
```

### What's Being Verified

Each E2E test:

1. **Generates spans** with unique test markers
2. **Forces flush** to send all data immediately
3. **Waits** for platform processing (5-20 seconds)
4. **Fetches traces** from the platform API
5. **Verifies** delivery rate (typically 90-95% threshold)

### Test Markers

Each span gets a unique `test_marker` attribute like `e2e-test-abc123`. This allows the test to identify its own traces even if other tests are running concurrently.

## Troubleshooting

### Issue: "JUDGMENT_API_KEY environment variable not set"

**Solution:** Set the environment variables as shown in Prerequisites.

### Issue: "pytest.skip: No traces found"

**Possible causes:**
1. Network issues - check connectivity to staging
2. API rate limiting - wait a few minutes and retry
3. Platform processing delay - tests wait up to 30s, but might need longer
4. Project limits reached - check staging dashboard

**Debug:**
```bash
# Check if traces are being sent
pytest src/tests/reliability/test_e2e_staging.py::TestE2ESustainedLoadSmall::test_e2e_100_spans_baseline -v -s

# The -s flag shows print statements including:
# - Project name/ID
# - Test marker
# - Number of spans generated
# - Number of traces found
```

### Issue: Tests timing out

**Solution:** Increase timeout:
```bash
pytest src/tests/reliability/test_e2e_staging.py -v --timeout=3600
```

### Issue: Assertion failures (delivery rate too low)

**Possible causes:**
1. Network instability
2. Platform under heavy load
3. Rate limiting
4. Bug in SDK export logic

**Debug:**
```python
# Add to test (temporary):
import pdb; pdb.set_trace()

# Then inspect:
print(f"Expected: {SPAN_COUNT}")
print(f"Received: {len(matching_traces)}")
print(f"Rate: {len(matching_traces) / SPAN_COUNT * 100:.1f}%")
```

### Issue: "Failed to resolve project"

**Possible causes:**
1. Invalid API key
2. Invalid organization ID
3. Insufficient permissions
4. Staging is down

**Debug:**
```bash
# Test project resolution directly
python -c "
from judgeval.v1.internal.api import JudgmentSyncClient
import os

client = JudgmentSyncClient(
    api_key=os.environ['JUDGMENT_API_KEY'],
    organization_id=os.environ['JUDGMENT_ORG_ID'],
    base_url=os.environ['JUDGMENT_API_URL']
)

response = client.projects_resolve(project_name='test-project')
print(response)
"
```

## Cleanup

### Manual Project Cleanup

E2E tests create new projects with names like `e2e-reliability-1234567890-abc123`.

To clean up old test projects:

```bash
# TODO: Add cleanup script once project deletion API is available
# python scripts/cleanup_e2e_projects.py --older-than 7d
```

For now, projects can be archived/deleted manually via the Judgment dashboard.

## Best Practices

### 1. Run Fast Tests First

Always run the fast tests before slow tests:

```bash
# Fast tests (1-2 minutes)
./run_e2e_tests.sh "e2e and not slow"

# If those pass, run slow tests (10-15 minutes)
./run_e2e_tests.sh "e2e and slow"
```

### 2. Check Staging Status

Before running tests, check if staging is healthy:

```bash
curl ${JUDGMENT_API_URL}/api/v1/health
```

### 3. Avoid Running During Deployments

Don't run E2E tests if staging is being deployed to. Check with the team first.

### 4. Use Unique Projects

Each test run creates a new project. This prevents interference between test runs but requires cleanup.

### 5. Monitor Rate Limits

If you're running tests frequently, you might hit rate limits. Space out test runs or request higher limits for test accounts.

## CI/CD Integration

E2E tests run automatically in CI:

- **Daily (2 AM):** Full E2E test suite runs against staging
- **Manual:** Can be triggered via GitHub Actions workflow dispatch

**View results:**
```
GitHub → Actions → Reliability Tests → e2e-tests job
```

## Next Steps

After verifying E2E tests work:

1. **Scale up:** Try larger test volumes (50k, 100k spans)
2. **Add metrics:** Integrate with Prometheus (see `PROMETHEUS_PLAN.md`)
3. **Add scenarios:** Test more edge cases (large payloads, many attributes, etc.)
4. **Automate cleanup:** Create script to delete old E2E test projects

## Reference

- Full test suite: [README.md](./README.md)
- Prometheus integration: [PROMETHEUS_PLAN.md](./PROMETHEUS_PLAN.md)
- Main reliability tests: `test_*.py` (mocked tests)
- E2E tests: `test_e2e_staging.py` (real API calls)
