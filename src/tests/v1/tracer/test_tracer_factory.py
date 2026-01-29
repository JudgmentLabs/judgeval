import pytest
from unittest.mock import MagicMock, patch
from judgeval.v1.tracer.tracer_factory import TracerFactory
from judgeval.v1.tracer.tracer import Tracer


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def tracer_factory(mock_client):
    return TracerFactory(
        client=mock_client,
        project_name="test_project",
        project_id="test_project_id",
    )


def test_factory_create_default(tracer_factory):
    tracer = tracer_factory.create()

    assert isinstance(tracer, Tracer)
    assert tracer.project_name == "test_project"
    assert tracer.project_id == "test_project_id"
    assert tracer.enable_evaluation is True
    assert tracer.enable_monitoring is True


def test_factory_create_with_custom_serializer(tracer_factory):
    def custom_serializer(x):
        return f"custom_{x}"

    tracer = tracer_factory.create(serializer=custom_serializer)

    assert isinstance(tracer, Tracer)


def test_factory_create_without_evaluation(tracer_factory):
    tracer = tracer_factory.create(enable_evaluation=False)

    assert tracer.enable_evaluation is False


def test_factory_create_without_monitoring(tracer_factory):
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider") as mock_set:
        tracer = tracer_factory.create(enable_monitoring=False)

        assert tracer.enable_monitoring is False
        mock_set.assert_not_called()


def test_factory_create_isolated(tracer_factory):
    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider") as mock_set:
        tracer = tracer_factory.create(isolated=True)

        assert tracer._tracer_provider is not None
        mock_set.assert_not_called()
