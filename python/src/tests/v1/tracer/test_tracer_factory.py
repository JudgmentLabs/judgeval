import pytest
from unittest.mock import MagicMock
from judgeval.v1.tracer.tracer_factory import TracerFactory
from judgeval.v1.tracer.tracer import Tracer


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def tracer_factory(mock_client):
    return TracerFactory(mock_client)


def test_factory_create_default(tracer_factory):
    tracer = tracer_factory.create(project_name="test_project")

    assert isinstance(tracer, Tracer)
    assert tracer.project_name == "test_project"
    assert tracer.enable_evaluation is True


def test_factory_create_with_custom_serializer(tracer_factory):
    def custom_serializer(x):
        return f"custom_{x}"

    tracer = tracer_factory.create(
        project_name="test_project", serializer=custom_serializer
    )

    assert isinstance(tracer, Tracer)


def test_factory_create_without_evaluation(tracer_factory):
    tracer = tracer_factory.create(project_name="test_project", enable_evaluation=False)

    assert tracer.enable_evaluation is False


def test_factory_create_with_initialize(tracer_factory):
    from unittest.mock import patch

    with patch("judgeval.v1.tracer.tracer.trace.set_tracer_provider"):
        tracer = tracer_factory.create(project_name="test_project", initialize=True)

        assert tracer._tracer_provider is not None
