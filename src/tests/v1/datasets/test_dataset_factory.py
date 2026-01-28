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
    return DatasetFactory(
        mock_client, project_id="test_project_id", project_name="test_project"
    )


@pytest.fixture
def sample_examples():
    return [
        Example(name="example1").set_property("input", "input1"),
        Example(name="example2").set_property("input", "input2"),
    ]


def test_factory_get(dataset_factory, mock_client):
    mock_client.get_projects_datasets_by_dataset_name.return_value = {
        "dataset_kind": "example",
        "examples": [],
    }

    dataset = dataset_factory.get("test_dataset")

    assert isinstance(dataset, Dataset)
    assert dataset.name == "test_dataset"
    assert dataset.project_id == "test_project_id"
    mock_client.get_projects_datasets_by_dataset_name.assert_called_once_with(
        project_id="test_project_id",
        dataset_name="test_dataset",
    )


def test_factory_create(dataset_factory, mock_client, sample_examples):
    dataset = dataset_factory.create(
        name="test_dataset",
        examples=sample_examples,
        overwrite=False,
    )

    assert isinstance(dataset, Dataset)
    assert dataset.name == "test_dataset"
    assert dataset.project_id == "test_project_id"
    assert len(dataset.examples) == 2
    mock_client.post_projects_datasets.assert_called_once_with(
        project_id="test_project_id",
        payload={
            "name": "test_dataset",
            "examples": [],
            "dataset_kind": "example",
            "overwrite": False,
        },
    )


def test_factory_create_with_overwrite(dataset_factory, mock_client):
    dataset = dataset_factory.create(name="test_dataset", examples=[], overwrite=True)

    assert isinstance(dataset, Dataset)
    mock_client.post_projects_datasets.assert_called_once_with(
        project_id="test_project_id",
        payload={
            "name": "test_dataset",
            "examples": [],
            "dataset_kind": "example",
            "overwrite": True,
        },
    )


def test_factory_list(dataset_factory, mock_client):
    mock_client.get_projects_datasets.return_value = [
        {
            "dataset_id": "1",
            "name": "dataset1",
            "created_at": "2024-01-01",
            "kind": "example",
            "entries": 10,
            "creator": "user1",
        }
    ]

    datasets = dataset_factory.list()

    assert isinstance(datasets, list)
    assert len(datasets) == 1
    assert isinstance(datasets[0], DatasetInfo)
    mock_client.get_projects_datasets.assert_called_once_with(
        project_id="test_project_id",
    )


def test_factory_create_empty_examples(dataset_factory, mock_client):
    dataset = dataset_factory.create(name="test_dataset")

    assert len(dataset.examples) == 0
