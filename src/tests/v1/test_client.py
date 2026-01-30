import pytest
from unittest.mock import patch
from judgeval.v1 import Judgeval
from judgeval.v1.datasets.dataset_factory import DatasetFactory
from judgeval.v1.evaluation.evaluation_factory import EvaluationFactory
from judgeval.v1.scorers.scorers_factory import ScorersFactory
from judgeval.v1.tracer.tracer_factory import TracerFactory
from judgeval.v1.trainers.trainers_factory import TrainersFactory


@pytest.fixture
def mock_resolve_project_id():
    with patch("judgeval.v1.resolve_project_id", return_value="test_project_id"):
        yield


def test_client_initialization_with_credentials(monkeypatch, mock_resolve_project_id):
    import uuid
    import judgeval.v1 as v1_module

    test_key = f"key_{uuid.uuid4()}"
    test_org = f"org_{uuid.uuid4()}"
    test_url = f"http://test_{uuid.uuid4()}.example.com"

    monkeypatch.setattr(v1_module, "JUDGMENT_API_KEY", test_key)
    monkeypatch.setattr(v1_module, "JUDGMENT_ORG_ID", test_org)
    monkeypatch.setattr(v1_module, "JUDGMENT_API_URL", test_url)

    client = Judgeval(project_name="test_project")

    assert client._api_key == test_key
    assert client._organization_id == test_org
    assert client._api_url == test_url


def test_client_initialization_with_explicit_credentials(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="explicit_key",
        organization_id="explicit_org",
        api_url="http://explicit.example.com",
    )

    assert client._api_key == "explicit_key"
    assert client._organization_id == "explicit_org"
    assert client._api_url == "http://explicit.example.com"


def test_client_missing_api_key(monkeypatch, mock_resolve_project_id):
    import judgeval.v1 as v1_module

    monkeypatch.setattr(v1_module, "JUDGMENT_API_KEY", None)
    monkeypatch.setattr(v1_module, "JUDGMENT_ORG_ID", "test_org")
    monkeypatch.setattr(v1_module, "JUDGMENT_API_URL", "http://test.example.com")

    with pytest.raises(ValueError, match="api_key is required"):
        Judgeval(project_name="test_project")


def test_client_missing_organization_id(monkeypatch, mock_resolve_project_id):
    import judgeval.v1 as v1_module

    monkeypatch.setattr(v1_module, "JUDGMENT_API_KEY", "test_key")
    monkeypatch.setattr(v1_module, "JUDGMENT_ORG_ID", None)
    monkeypatch.setattr(v1_module, "JUDGMENT_API_URL", "http://test.example.com")

    with pytest.raises(ValueError, match="organization_id is required"):
        Judgeval(project_name="test_project")


def test_client_api_url_default(monkeypatch, mock_resolve_project_id):
    import uuid
    import judgeval.v1 as v1_module

    monkeypatch.setattr(v1_module, "JUDGMENT_API_KEY", f"key_{uuid.uuid4()}")
    monkeypatch.setattr(v1_module, "JUDGMENT_ORG_ID", f"org_{uuid.uuid4()}")
    monkeypatch.setattr(v1_module, "JUDGMENT_API_URL", "https://api.judgmentlabs.ai")

    client = Judgeval(project_name="test_project")

    assert client._api_url == "https://api.judgmentlabs.ai"


def test_client_tracer_factory_property(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="test_key",
        organization_id="test_org",
        api_url="http://test.com",
    )
    tracer_factory = client.tracer
    assert isinstance(tracer_factory, TracerFactory)


def test_client_scorers_factory_property(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="test_key",
        organization_id="test_org",
        api_url="http://test.com",
    )
    scorers_factory = client.scorers
    assert isinstance(scorers_factory, ScorersFactory)


def test_client_evaluation_factory_property(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="test_key",
        organization_id="test_org",
        api_url="http://test.com",
    )
    evaluation_factory = client.evaluation
    assert isinstance(evaluation_factory, EvaluationFactory)


def test_client_trainers_factory_property(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="test_key",
        organization_id="test_org",
        api_url="http://test.com",
    )
    trainers_factory = client.trainers
    assert isinstance(trainers_factory, TrainersFactory)


def test_client_datasets_factory_property(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="test_key",
        organization_id="test_org",
        api_url="http://test.com",
    )
    datasets_factory = client.datasets
    assert isinstance(datasets_factory, DatasetFactory)
