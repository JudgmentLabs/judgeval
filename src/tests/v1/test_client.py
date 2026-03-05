import pytest
from unittest.mock import patch
from judgeval.v1 import Judgeval
from judgeval.v1.datasets.dataset_factory import DatasetFactory
from judgeval.v1.evaluation.evaluation_factory import EvaluationFactory
from judgeval.v1.judges.judges_factory import JudgesFactory


@pytest.fixture
def mock_resolve_project_id():
    with patch(
        "judgeval.v1.judgeval.resolve_project_id", return_value="test_project_id"
    ):
        yield


def test_client_initialization_with_credentials(monkeypatch, mock_resolve_project_id):
    import uuid
    import judgeval.v1.judgeval as judgeval_mod

    test_key = f"key_{uuid.uuid4()}"
    test_org = f"org_{uuid.uuid4()}"
    test_url = f"http://test_{uuid.uuid4()}.example.com"

    monkeypatch.setattr(judgeval_mod, "JUDGMENT_API_KEY", test_key)
    monkeypatch.setattr(judgeval_mod, "JUDGMENT_ORG_ID", test_org)
    monkeypatch.setattr(judgeval_mod, "JUDGMENT_API_URL", test_url)

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
    import judgeval.v1.judgeval as judgeval_mod

    monkeypatch.setattr(judgeval_mod, "JUDGMENT_API_KEY", None)
    monkeypatch.setattr(judgeval_mod, "JUDGMENT_ORG_ID", "test_org")
    monkeypatch.setattr(judgeval_mod, "JUDGMENT_API_URL", "http://test.example.com")

    with pytest.raises(ValueError, match="api_key is required"):
        Judgeval(project_name="test_project")


def test_client_missing_organization_id(monkeypatch, mock_resolve_project_id):
    import judgeval.v1.judgeval as judgeval_mod

    monkeypatch.setattr(judgeval_mod, "JUDGMENT_API_KEY", "test_key")
    monkeypatch.setattr(judgeval_mod, "JUDGMENT_ORG_ID", None)
    monkeypatch.setattr(judgeval_mod, "JUDGMENT_API_URL", "http://test.example.com")

    with pytest.raises(ValueError, match="organization_id is required"):
        Judgeval(project_name="test_project")


def test_client_api_url_default(monkeypatch, mock_resolve_project_id):
    import uuid
    import judgeval.v1.judgeval as judgeval_mod

    monkeypatch.setattr(judgeval_mod, "JUDGMENT_API_KEY", f"key_{uuid.uuid4()}")
    monkeypatch.setattr(judgeval_mod, "JUDGMENT_ORG_ID", f"org_{uuid.uuid4()}")
    monkeypatch.setattr(judgeval_mod, "JUDGMENT_API_URL", "https://api.judgmentlabs.ai")

    client = Judgeval(project_name="test_project")

    assert client._api_url == "https://api.judgmentlabs.ai"


def test_client_judges_factory_property(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="test_key",
        organization_id="test_org",
        api_url="http://test.com",
    )
    judges_factory = client.judges
    assert isinstance(judges_factory, JudgesFactory)


def test_client_evaluation_factory_property(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="test_key",
        organization_id="test_org",
        api_url="http://test.com",
    )
    evaluation_factory = client.evaluation
    assert isinstance(evaluation_factory, EvaluationFactory)


def test_client_datasets_factory_property(mock_resolve_project_id):
    client = Judgeval(
        project_name="test_project",
        api_key="test_key",
        organization_id="test_org",
        api_url="http://test.com",
    )
    datasets_factory = client.datasets
    assert isinstance(datasets_factory, DatasetFactory)


def test_client_project_not_found_logs_warning(caplog):
    import logging

    with patch("judgeval.v1.judgeval.resolve_project_id", return_value=None):
        with caplog.at_level(logging.WARNING):
            client = Judgeval(
                project_name="nonexistent_project",
                api_key="test_key",
                organization_id="test_org",
                api_url="http://test.com",
            )

    assert client._project_id is None
    assert "nonexistent_project" in caplog.text
    assert "not found" in caplog.text


def test_client_project_not_found_still_creates_factories():
    with patch("judgeval.v1.judgeval.resolve_project_id", return_value=None):
        client = Judgeval(
            project_name="nonexistent_project",
            api_key="test_key",
            organization_id="test_org",
            api_url="http://test.com",
        )

    assert client._project_id is None
    assert isinstance(client.judges, JudgesFactory)
    assert isinstance(client.evaluation, EvaluationFactory)
    assert isinstance(client.datasets, DatasetFactory)
