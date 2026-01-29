"""
End-to-end tests against staging environment.

These tests verify that traces actually make it to the platform
with correct data integrity. They run against a real API endpoint.

Prerequisites:
    export JUDGMENT_API_KEY="your-api-key"
    export JUDGMENT_ORG_ID="your-org-id"
    export JUDGMENT_API_URL="https://staging.judgmentlabs.ai"

Run with:
    uv run pytest src/tests/reliability/test_e2e_staging.py -v -s -m "e2e and not slow"
    uv run pytest src/tests/reliability/test_e2e_staging.py -v -s -k "test_e2e_100_spans_baseline"
"""

import os
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlencode

import pytest

try:
    import aiohttp
except ImportError:
    pytest.skip(
        "aiohttp not installed, run: pip install aiohttp", allow_module_level=True
    )

from judgeval import Judgeval
from judgeval.v1.tracer.tracer import Tracer


# ============================================================================
# Trace Fetching Client
# ============================================================================


@dataclass(frozen=True)
class TracesClientConfig:
    access_token: str
    organization_id: str
    base_url: str = "https://staging.judgmentlabs.ai"


class TracesClient:
    """Async client for fetching traces from the Judgment API."""

    def __init__(self, config: TracesClientConfig):
        self._base_url = config.base_url.rstrip("/")
        self._access_token = config.access_token
        self._org_id = config.organization_id

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Authorization": f"Bearer {self._access_token}",
            "X-Organization-Id": self._org_id,
        }

    def _build_url(
        self,
        project_id: str,
        limit: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> str:
        params: Dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time

        path = f"/projects/{project_id}/traces"
        if params:
            return f"{self._base_url}{path}?{urlencode(params)}"

        url = f"{self._base_url}{path}"
        print(f"Fetching traces from: {url=}")
        return url

    async def get_traces(
        self,
        project_id: str,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch traces from the API."""
        url = self._build_url(
            project_id=project_id,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )
        print(f"Fetching traces from: {url=}")

        body = {"filters": []}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._headers(), json=body) as resp:
                if resp.status == 401:
                    raise RuntimeError("Unauthorized: Invalid or expired access token")
                if resp.status == 403:
                    raise RuntimeError("Forbidden: Insufficient permissions")
                if resp.status == 404:
                    raise RuntimeError(
                        f"Not Found: Project {project_id} does not exist"
                    )
                if resp.status < 200 or resp.status >= 300:
                    text = await resp.text()
                    raise RuntimeError(
                        f"API request failed: {resp.status} {resp.reason} - {text}"
                    )

                data = await resp.json()
                return data.get("data", [])


def iso_now_minus(minutes: int) -> str:
    """Get ISO timestamp N minutes ago."""
    return (datetime.utcnow() - timedelta(minutes=minutes)).replace(
        microsecond=0
    ).isoformat() + "Z"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def staging_config() -> Dict[str, str]:
    """Get staging environment configuration from environment variables."""
    api_key = os.environ.get("JUDGMENT_API_KEY")
    org_id = os.environ.get("JUDGMENT_ORG_ID")
    api_url = os.environ.get("JUDGMENT_API_URL", "https://staging.judgmentlabs.ai")

    if not api_key:
        pytest.skip("JUDGMENT_API_KEY environment variable not set")

    if not org_id:
        pytest.skip("JUDGMENT_ORG_ID environment variable not set")

    return {
        "api_key": api_key,
        "org_id": org_id,
        "api_url": api_url.rstrip("/"),
    }


@pytest.fixture(scope="module")
def judgeval_client(staging_config: Dict[str, str]) -> Judgeval:
    """Create Judgeval client (uses public API)."""
    # Judgeval() reads from environment variables automatically
    return Judgeval()


@pytest.fixture(scope="module")
def traces_client(staging_config: Dict[str, str]) -> TracesClient:
    """Create traces fetching client."""
    return TracesClient(
        TracesClientConfig(
            access_token=staging_config["api_key"],
            organization_id=staging_config["org_id"],
            base_url=staging_config["api_url"],
        )
    )


@pytest.fixture
def unique_project_name() -> str:
    """Generate unique project name for each test."""
    timestamp = int(time.time())
    random_id = uuid.uuid4().hex[:8]
    project_name = f"e2e-reliability-{timestamp}-{random_id}"

    print(f"\n[E2E] Created project: {project_name}")

    return project_name


@pytest.fixture
def e2e_tracer(judgeval_client: Judgeval, unique_project_name: str) -> Tracer:
    """Create a tracer configured for E2E testing against staging."""
    print(f"[E2E] Creating tracer for project: {unique_project_name}")

    # Use the public API
    tracer = judgeval_client.tracer.create(project_name=unique_project_name)

    return tracer


@pytest.fixture
def project_id(e2e_tracer: Tracer) -> str:
    """Get project ID from the tracer."""
    project_id = e2e_tracer.project_id

    if not project_id:
        pytest.fail("Failed to resolve project_id from tracer")

    print(f"[E2E] Resolved project_id: {project_id}")
    return project_id


async def fetch_traces_with_retry(
    traces_client: TracesClient,
    project_id: str,
    test_marker: str,
    max_retries: int = 15,
    retry_delay: float = 2.0,
    lookback_minutes: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch traces from platform with retry logic.

    Args:
        traces_client: TracesClient instance
        project_id: Project ID to fetch traces from
        test_marker: Unique test marker to filter traces
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        lookback_minutes: How many minutes back to search

    Returns:
        List of matching trace dictionaries
    """
    start_time = iso_now_minus(lookback_minutes)

    for attempt in range(max_retries):
        try:
            all_traces = await traces_client.get_traces(
                project_id=project_id,
                limit=10000,
                start_time=start_time,
            )

            # Filter by test marker in span_name
            matching_traces = []
            for trace in all_traces:
                span_name = trace.get("span_name", "")
                if test_marker in span_name:
                    matching_traces.append(trace)

            if matching_traces:
                print(
                    f"[E2E] Found {len(matching_traces)} matching traces "
                    f"(out of {len(all_traces)} total)"
                )
                return matching_traces

            # No matches yet, retry
            if attempt < max_retries - 1:
                print(
                    f"[E2E] No matching traces found, retrying in {retry_delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[E2E] Error fetching traces: {e}, retrying...")
                await asyncio.sleep(retry_delay)
            else:
                raise

    print(f"[E2E] Warning: No matching traces found after {max_retries} attempts")
    return []


# ============================================================================
# E2E Sustained Load Tests - Small Scale
# ============================================================================


@pytest.mark.e2e
@pytest.mark.reliability
class TestE2ESustainedLoadSmall:
    """Small-scale E2E sustained load tests."""

    def test_e2e_100_spans_baseline(
        self,
        e2e_tracer: Tracer,
        traces_client: TracesClient,
        project_id: str,
    ):
        """
        Baseline test: Generate 100 spans and verify they reach the platform.
        """
        SPAN_COUNT = 100
        TEST_MARKER = f"e2e-test-{uuid.uuid4().hex[:8]}"

        print(f"\n[E2E] Starting baseline test with {SPAN_COUNT} spans")
        print(f"[E2E] Test marker: {TEST_MARKER}")

        # Generate spans using public API
        @e2e_tracer.observe(span_type="function")
        def generate_span(i: int):
            """Generate a single test span."""
            e2e_tracer.set_attribute("span_index", i)
            e2e_tracer.set_attribute("total_spans", SPAN_COUNT)
            return f"result-{i}"

        # Generate spans
        for i in range(SPAN_COUNT):
            # Use span name with test marker
            with e2e_tracer.span(f"baseline-{TEST_MARKER}-{i}"):
                generate_span(i)

        # Force flush
        print(f"[E2E] Generated {SPAN_COUNT} spans, forcing flush...")
        flush_success = e2e_tracer.force_flush(timeout_millis=30000)
        assert flush_success, "Force flush timed out"

        # Wait for platform processing
        print("[E2E] Waiting 5s for platform processing...")
        time.sleep(5)

        # Fetch traces
        matching_traces = asyncio.run(
            fetch_traces_with_retry(
                traces_client,
                project_id,
                TEST_MARKER,
                max_retries=20,
                retry_delay=2.0,
            )
        )

        # Verify delivery (allow 95% success rate)
        expected_minimum = int(SPAN_COUNT * 0.95)
        assert len(matching_traces) >= expected_minimum, (
            f"Expected at least {expected_minimum} traces (95%), "
            f"got {len(matching_traces)}"
        )

    def test_e2e_1k_spans_over_10_seconds(
        self,
        e2e_tracer: Tracer,
        traces_client: TracesClient,
        project_id: str,
    ):
        """Generate 1,000 spans over 10 seconds (~100 spans/sec)."""
        SPAN_COUNT = 1_000
        DURATION_SECONDS = 10
        TEST_MARKER = f"e2e-test-{uuid.uuid4().hex[:8]}"

        print(f"\n[E2E] Starting 1k span test over {DURATION_SECONDS}s")
        print(f"[E2E] Test marker: {TEST_MARKER}")

        # Generate spans with pacing
        start_time = time.perf_counter()
        spans_per_second = SPAN_COUNT / DURATION_SECONDS

        for i in range(SPAN_COUNT):
            with e2e_tracer.span(f"paced-{TEST_MARKER}-{i}"):
                e2e_tracer.set_attribute("span_index", i)
                e2e_tracer.set_attribute("timestamp", time.time())

            # Pace generation
            elapsed = time.perf_counter() - start_time
            expected_elapsed = (i + 1) / spans_per_second
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)

        actual_duration = time.perf_counter() - start_time
        actual_rate = SPAN_COUNT / actual_duration

        print(
            f"[E2E] Generated {SPAN_COUNT} spans in {actual_duration:.1f}s "
            f"({actual_rate:.0f} spans/sec)"
        )

        # Force flush
        print("[E2E] Forcing flush...")
        flush_success = e2e_tracer.force_flush(timeout_millis=30000)
        assert flush_success, "Force flush timed out"

        # Wait for processing
        print("[E2E] Waiting 8s for platform processing...")
        time.sleep(8)

        # Verify delivery
        matching_traces = asyncio.run(
            fetch_traces_with_retry(
                traces_client,
                project_id,
                TEST_MARKER,
                max_retries=25,
                retry_delay=2.0,
                lookback_minutes=10,
            )
        )

        expected_minimum = int(SPAN_COUNT * 0.95)
        assert len(matching_traces) >= expected_minimum, (
            f"Expected at least {expected_minimum} traces (95%), "
            f"got {len(matching_traces)}"
        )


@pytest.mark.e2e
@pytest.mark.reliability
@pytest.mark.slow
class TestE2ESustainedLoadMedium:
    """Medium-scale E2E sustained load tests."""

    def test_e2e_10k_spans_over_30_seconds(
        self,
        e2e_tracer: Tracer,
        traces_client: TracesClient,
        project_id: str,
    ):
        """Generate 10,000 spans over 30 seconds (~333 spans/sec)."""
        SPAN_COUNT = 10_000
        DURATION_SECONDS = 30
        TEST_MARKER = f"e2e-test-{uuid.uuid4().hex[:8]}"

        print(f"\n[E2E] Starting 10k span test over {DURATION_SECONDS}s")
        print(f"[E2E] Test marker: {TEST_MARKER}")

        # Generate spans with pacing
        start_time = time.perf_counter()
        spans_per_second = SPAN_COUNT / DURATION_SECONDS

        for i in range(SPAN_COUNT):
            with e2e_tracer.span(f"medium-{TEST_MARKER}-{i}"):
                e2e_tracer.set_attribute("span_index", i)

            # Pace generation
            elapsed = time.perf_counter() - start_time
            expected_elapsed = (i + 1) / spans_per_second
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)

            # Progress logging
            if (i + 1) % 2000 == 0:
                current_elapsed = time.perf_counter() - start_time
                current_rate = (i + 1) / current_elapsed
                print(
                    f"[E2E] Progress: {i + 1}/{SPAN_COUNT} spans "
                    f"({current_rate:.0f} spans/sec)"
                )

        actual_duration = time.perf_counter() - start_time
        actual_rate = SPAN_COUNT / actual_duration

        print(
            f"[E2E] Generated {SPAN_COUNT} spans in {actual_duration:.1f}s "
            f"({actual_rate:.0f} spans/sec)"
        )

        # Force flush with longer timeout
        print("[E2E] Forcing flush (may take a while)...")
        flush_success = e2e_tracer.force_flush(timeout_millis=60000)
        assert flush_success, "Force flush timed out after 60s"

        # Wait for processing
        print("[E2E] Waiting 15s for platform processing...")
        time.sleep(15)

        # Verify delivery (90% threshold for larger volume)
        matching_traces = asyncio.run(
            fetch_traces_with_retry(
                traces_client,
                project_id,
                TEST_MARKER,
                max_retries=30,
                retry_delay=3.0,
                lookback_minutes=15,
            )
        )

        expected_minimum = int(SPAN_COUNT * 0.90)
        assert len(matching_traces) >= expected_minimum, (
            f"Expected at least {expected_minimum} traces (90%), "
            f"got {len(matching_traces)}"
        )


# ============================================================================
# Run single test for quick verification
# ============================================================================

if __name__ == "__main__":
    # Quick test run
    pytest.main([__file__, "-v", "-m", "e2e and not slow", "-s"])
