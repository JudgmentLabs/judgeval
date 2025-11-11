import pytest
from unittest.mock import MagicMock, patch
from judgeval.v1.tracer.tracer import Tracer
from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def serializer():
    return lambda x: str(x)


def test_tracer_initialization(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        api_client=mock_client,
        serializer=serializer,
        initialize=False,
    )

    assert tracer.project_name == "test_project"
    assert tracer._tracer_provider is None


def test_tracer_initialization_with_initialize(mock_client, serializer):
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider"):
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=True,
            api_client=mock_client,
            serializer=serializer,
            initialize=True,
        )

        assert tracer._tracer_provider is not None
        assert isinstance(tracer._tracer_provider, TracerProvider)


def test_tracer_force_flush_without_initialization(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        api_client=mock_client,
        serializer=serializer,
        initialize=False,
    )

    result = tracer.force_flush()
    assert result is False


def test_tracer_force_flush_with_initialization(mock_client, serializer):
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider"):
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=True,
            api_client=mock_client,
            serializer=serializer,
            initialize=True,
        )

        tracer._tracer_provider.force_flush = MagicMock(return_value=True)
        result = tracer.force_flush(timeout_millis=5000)

        assert result is True
        tracer._tracer_provider.force_flush.assert_called_once_with(5000)


def test_tracer_shutdown_without_initialization(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        api_client=mock_client,
        serializer=serializer,
        initialize=False,
    )

    tracer.shutdown()


def test_tracer_shutdown_with_initialization(mock_client, serializer):
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider"):
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=True,
            api_client=mock_client,
            serializer=serializer,
            initialize=True,
        )

        tracer._tracer_provider.shutdown = MagicMock()
        tracer.shutdown(timeout_millis=10000)

        tracer._tracer_provider.shutdown.assert_called_once()
