import pytest
from unittest.mock import MagicMock
from judgeval.v1.datasets.dataset_factory import DatasetFactory
from judgeval.v1.datasets.dataset import Dataset, DatasetInfo
from judgeval.v1.data.example import Example


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def dataset_factory(mock_client):
    return DatasetFactory(mock_client)


@pytest.fixture
def sample_examples():
    return [
        Example(name="example1").set_property("input", "input1"),
        Example(name="example2").set_property("input", "input2"),
    ]


def test_factory_get(dataset_factory, mock_client):
    mock_client.datasets_pull_for_judgeval.return_value = {
        "dataset_kind": "example",
        "examples": [],
    }

    dataset = dataset_factory.get("test_dataset", "test_project")

    assert isinstance(dataset, Dataset)
    assert dataset.name == "test_dataset"
    assert dataset.project_name == "test_project"
    mock_client.datasets_pull_for_judgeval.assert_called_once()


def test_factory_create(dataset_factory, mock_client, sample_examples):
    dataset = dataset_factory.create(
        name="test_dataset",
        project_name="test_project",
        examples=sample_examples,
        overwrite=False,
    )

    assert isinstance(dataset, Dataset)
    assert dataset.name == "test_dataset"
    assert dataset.project_name == "test_project"
    assert len(dataset.examples) == 2
    mock_client.datasets_create_for_judgeval.assert_called_once()


def test_factory_create_with_overwrite(dataset_factory, mock_client):
    dataset = dataset_factory.create(
        name="test_dataset", project_name="test_project", examples=[], overwrite=True
    )

    assert isinstance(dataset, Dataset)
    call_args = mock_client.datasets_create_for_judgeval.call_args[0][0]
    assert call_args["overwrite"] is True


def test_factory_list(dataset_factory, mock_client):
    mock_client.datasets_pull_all_for_judgeval.return_value = [
        {
            "dataset_id": "1",
            "name": "dataset1",
            "created_at": "2024-01-01",
            "kind": "example",
            "entries": 10,
            "creator": "user1",
        }
    ]

    datasets = dataset_factory.list("test_project")

    assert isinstance(datasets, list)
    assert len(datasets) == 1
    assert isinstance(datasets[0], DatasetInfo)
    mock_client.datasets_pull_all_for_judgeval.assert_called_once()


def test_factory_create_empty_examples(dataset_factory, mock_client):
    dataset = dataset_factory.create(name="test_dataset", project_name="test_project")

    assert len(dataset.examples) == 0
