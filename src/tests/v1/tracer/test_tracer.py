import pytest
from unittest.mock import MagicMock, patch
from judgeval.v1.tracer.tracer import Tracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import context as otel_context


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
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
    )

    assert tracer.project_name == "test_project"
    assert tracer._tracer_provider is not None
    assert isinstance(tracer._tracer_provider, TracerProvider)


def test_tracer_initialization_monitoring_disabled(mock_client, serializer):
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider") as mock_set:
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=True,
            enable_monitoring=False,
            api_client=mock_client,
            serializer=serializer,
        )

        assert tracer._tracer_provider is not None
        mock_set.assert_not_called()


def test_tracer_initialization_isolated(mock_client, serializer):
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider") as mock_set:
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=True,
            enable_monitoring=True,
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        assert tracer._tracer_provider is not None
        mock_set.assert_not_called()


def test_tracer_force_flush(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
    )

    tracer._tracer_provider.force_flush = MagicMock(return_value=True)
    result = tracer.force_flush(timeout_millis=5000)

    assert result is True
    tracer._tracer_provider.force_flush.assert_called_once_with(5000)


def test_tracer_shutdown(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
    )

    tracer._tracer_provider.shutdown = MagicMock()
    tracer.shutdown(timeout_millis=10000)

    tracer._tracer_provider.shutdown.assert_called_once()


def test_tracer_get_context_non_isolated(mock_client, serializer):
    """Test that get_context returns global context when not in isolated mode."""
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
        isolated=False,
    )

    # Get the global context
    global_context = otel_context.get_current()
    tracer_context = tracer.get_context()

    # They should be the same object
    assert tracer_context is global_context


def test_tracer_get_context_isolated(mock_client, serializer):
    """Test that get_context returns isolated context when in isolated mode."""
    from judgeval.v1.tracer.isolated.context import get_current as get_isolated_context

    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
        isolated=True,
    )

    # Get the isolated context
    isolated_context = get_isolated_context()
    tracer_context = tracer.get_context()

    # They should be the same object
    assert tracer_context is isolated_context

    # And it should NOT be the global context
    global_context = otel_context.get_current()
    assert tracer_context is not global_context
