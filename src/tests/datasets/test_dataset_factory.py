from __future__ import annotations

from unittest.mock import MagicMock


from judgeval.datasets.dataset import Dataset, DatasetInfo
from judgeval.datasets.dataset_factory import DatasetFactory


def _make_factory(project_id="proj-1"):
    client = MagicMock()
    return DatasetFactory(
        client=client, project_id=project_id, project_name="test-project"
    ), client


class TestDatasetFactoryGet:
    def test_get_returns_dataset(self):
        factory, client = _make_factory()
        client.get_projects_datasets_by_dataset_name.return_value = {
            "dataset_kind": "example",
            "examples": [],
        }
        ds = factory.get("my-dataset")
        assert isinstance(ds, Dataset)
        assert ds.name == "my-dataset"

    def test_get_maps_examples(self):
        factory, client = _make_factory()
        client.get_projects_datasets_by_dataset_name.return_value = {
            "dataset_kind": "example",
            "examples": [
                {
                    "example_id": "e1",
                    "created_at": "2024-01-01",
                    "name": "ex",
                    "input": "q",
                }
            ],
        }
        ds = factory.get("ds")
        assert len(ds) == 1
        assert ds.examples[0]["input"] == "q"

    def test_get_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.get("ds")
        assert result is None


class TestDatasetFactoryCreate:
    def test_create_returns_dataset(self):
        factory, client = _make_factory()
        client.post_projects_datasets.return_value = {}
        client.post_projects_datasets_by_dataset_name_examples.return_value = {}
        ds = factory.create("new-ds")
        assert isinstance(ds, Dataset)
        assert ds.name == "new-ds"

    def test_create_calls_post_datasets(self):
        factory, client = _make_factory()
        client.post_projects_datasets.return_value = {}
        factory.create("new-ds")
        client.post_projects_datasets.assert_called_once()

    def test_create_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.create("ds")
        assert result is None


class TestDatasetFactoryList:
    def test_list_returns_dataset_infos(self):
        factory, client = _make_factory()
        client.get_projects_datasets.return_value = [
            {
                "dataset_id": "d1",
                "name": "ds1",
                "created_at": "2024-01-01",
                "kind": "example",
                "entries": 5.0,
                "creator": "user",
            }
        ]
        result = factory.list()
        assert len(result) == 1
        assert isinstance(result[0], DatasetInfo)
        assert result[0].name == "ds1"

    def test_list_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.list()
        assert result is None
