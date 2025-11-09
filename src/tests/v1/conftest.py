from typing import Optional
import pytest
import random
import string
from unittest.mock import MagicMock
from opentelemetry.sdk.trace import ReadableSpan
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.data.example import Example


@pytest.fixture
def mock_client():
    client = MagicMock(spec=JudgmentSyncClient)
    client.api_key = "test_key"
    client.organization_id = "test_org"
    client.base_url = "http://test.com"
    return client


@pytest.fixture
def create_mock_span():
    def _create(trace_id: int, span_id: Optional[int] = None):
        span = MagicMock(spec=ReadableSpan)
        context = MagicMock()
        context.trace_id = trace_id
        context.span_id = span_id or trace_id
        span.get_span_context.return_value = context
        return span

    return _create


@pytest.fixture
def sample_examples():
    return [
        Example(name="ex1").create(input="q1", output="a1"),
        Example(name="ex2").create(input="q2", output="a2"),
    ]


@pytest.fixture
def project_name():
    return "test-project-" + "".join(
        random.choices(string.ascii_letters + string.digits, k=8)
    )


@pytest.fixture
def random_name():
    return "".join(random.choices(string.ascii_letters + string.digits, k=12))


@pytest.fixture
def eval_run_name():
    return "eval-run-" + "".join(
        random.choices(string.ascii_letters + string.digits, k=8)
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests for v1 API")
    config.addinivalue_line("markers", "integration: integration tests for v1 API")
