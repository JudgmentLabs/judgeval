import warnings
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
    assert tracer.enable_monitoring is True


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


def test_factory_create_without_monitoring(tracer_factory):
    tracer = tracer_factory.create(project_name="test_project", enable_monitoring=False)

    assert tracer.enable_monitoring is False


def test_factory_deprecated_isolated_param_emits_warning(tracer_factory):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tracer_factory.create(project_name="test_project", isolated=False)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "isolated" in str(w[0].message)


def test_factory_deprecated_initialize_param_emits_warning(tracer_factory):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tracer_factory.create(project_name="test_project", initialize=False)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "initialize" in str(w[0].message)
