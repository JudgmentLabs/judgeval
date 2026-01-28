"""
Shared fixtures for reliability tests.

These fixtures provide mocked v1 SDK components for testing
without requiring actual API credentials or network calls.
"""

import pytest
from typing import Optional
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import ReadableSpan

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.tracer.tracer_factory import TracerFactory
from judgeval.v1.tracer.tracer import Tracer


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "reliability: mark test as reliability/sanity check test"
    )


@pytest.fixture
def mock_client() -> MagicMock:
    """
    Create a mocked JudgmentSyncClient.

    This mock simulates successful API responses without making
    actual network calls.
    """
    client = MagicMock(spec=JudgmentSyncClient)
    client.api_key = "test_api_key"
    client.organization_id = "test_org_id"
    client.base_url = "http://test.judgmentlabs.ai/"

    # Mock successful project resolution
    client.projects_resolve.return_value = {"project_id": "test_project_id"}

    # Mock successful evaluation queue
    client.add_to_run_eval_queue_examples.return_value = {"success": True}

    return client


@pytest.fixture
def mock_client_with_timeout() -> MagicMock:
    """
    Create a mocked client that simulates timeout errors.
    """
    import httpx

    client = MagicMock(spec=JudgmentSyncClient)
    client.api_key = "test_api_key"
    client.organization_id = "test_org_id"
    client.base_url = "http://test.judgmentlabs.ai/"

    # Mock project resolution to succeed
    client.projects_resolve.return_value = {"project_id": "test_project_id"}

    # Mock evaluation to timeout
    client.add_to_run_eval_queue_examples.side_effect = httpx.TimeoutException(
        "timeout"
    )

    return client


@pytest.fixture
def mock_client_with_error() -> MagicMock:
    """
    Create a mocked client that simulates HTTP 500 errors.
    """
    client = MagicMock(spec=JudgmentSyncClient)
    client.api_key = "test_api_key"
    client.organization_id = "test_org_id"
    client.base_url = "http://test.judgmentlabs.ai/"

    # Mock project resolution to succeed
    client.projects_resolve.return_value = {"project_id": "test_project_id"}

    # Mock evaluation to fail with HTTP 500
    from judgeval.exceptions import JudgmentAPIError

    client.add_to_run_eval_queue_examples.side_effect = JudgmentAPIError(
        500, "Internal Server Error", None
    )

    return client


@pytest.fixture
def mock_client_project_failure() -> MagicMock:
    """
    Create a mocked client that fails to resolve project.
    """
    client = MagicMock(spec=JudgmentSyncClient)
    client.api_key = "test_api_key"
    client.organization_id = "test_org_id"
    client.base_url = "http://test.judgmentlabs.ai/"

    # Mock project resolution to fail
    client.projects_resolve.return_value = {"project_id": None}

    return client


@pytest.fixture
def tracer(mock_client: MagicMock) -> Tracer:
    """
    Create a v1 tracer with mocked client and monitoring enabled.

    This tracer is configured for testing without making actual API calls.
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="test_project_id"):
        factory = TracerFactory(mock_client)
        return factory.create(
            project_name="reliability-test",
            enable_monitoring=True,
            enable_evaluation=False,
            isolated=True,  # Use isolated mode to avoid global state pollution
        )


@pytest.fixture
def tracer_with_evaluation(mock_client: MagicMock) -> Tracer:
    """
    Create a v1 tracer with both monitoring and evaluation enabled.
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="test_project_id"):
        factory = TracerFactory(mock_client)
        return factory.create(
            project_name="reliability-test-eval",
            enable_monitoring=True,
            enable_evaluation=True,
            isolated=True,
        )


@pytest.fixture
def tracer_monitoring_disabled(mock_client: MagicMock) -> Tracer:
    """
    Create a v1 tracer with monitoring disabled.

    This is useful for testing graceful degradation.
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="test_project_id"):
        factory = TracerFactory(mock_client)
        return factory.create(
            project_name="reliability-test-disabled",
            enable_monitoring=False,
            enable_evaluation=False,
            isolated=True,
        )


@pytest.fixture
def create_mock_span():
    """
    Factory fixture to create mock ReadableSpan objects.
    """

    def _create(trace_id: int, span_id: Optional[int] = None) -> MagicMock:
        span = MagicMock(spec=ReadableSpan)
        context = MagicMock()
        context.trace_id = trace_id
        context.span_id = span_id or trace_id
        context.is_valid = True
        context.trace_flags = MagicMock()
        context.trace_flags.sampled = True
        span.get_span_context.return_value = context
        return span

    return _create
