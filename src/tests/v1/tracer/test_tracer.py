import pytest
from typing import Any
from unittest.mock import MagicMock, patch
from judgeval.v1.tracer.tracer import Tracer
from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.organization_id = "test_org"
    client.base_url = "http://test.com/"
    return client


@pytest.fixture
def serializer() -> Any:
    def serialize(x: object) -> str:
        return str(x)

    return serialize


def test_tracer_initialization(mock_client: MagicMock, serializer: Any) -> None:
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        api_client=mock_client,
        serializer=serializer,
        initialize=False,
    )

    assert tracer.project_name == "test_project"
    assert tracer._tracer_provider is None


def test_tracer_initialization_with_initialize(
    mock_client: MagicMock, serializer: Any
) -> None:
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider"):
        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            tracer = Tracer(
                project_name="test_project",
                enable_evaluation=True,
                api_client=mock_client,
                serializer=serializer,
                initialize=True,
            )

            assert tracer._tracer_provider is not None
            assert isinstance(tracer._tracer_provider, TracerProvider)


def test_tracer_force_flush_without_initialization(
    mock_client: MagicMock, serializer: Any
) -> None:
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        api_client=mock_client,
        serializer=serializer,
        initialize=False,
    )

    result = tracer.force_flush()
    assert result is False


def test_tracer_force_flush_with_initialization(
    mock_client: MagicMock, serializer: Any
) -> None:
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider"):
        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            tracer = Tracer(
                project_name="test_project",
                enable_evaluation=True,
                api_client=mock_client,
                serializer=serializer,
                initialize=True,
            )

            assert tracer._tracer_provider is not None
            result = tracer.force_flush(timeout_millis=5000)
            assert isinstance(result, bool)


def test_tracer_shutdown_without_initialization(
    mock_client: MagicMock, serializer: Any
) -> None:
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        api_client=mock_client,
        serializer=serializer,
        initialize=False,
    )

    tracer.shutdown()


def test_tracer_shutdown_with_initialization(
    mock_client: MagicMock, serializer: Any
) -> None:
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider"):
        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            tracer = Tracer(
                project_name="test_project",
                enable_evaluation=True,
                api_client=mock_client,
                serializer=serializer,
                initialize=True,
            )

            assert tracer._tracer_provider is not None
            tracer.shutdown(timeout_millis=10000)
