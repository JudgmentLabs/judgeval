from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from judgeval.data.example import Example
from judgeval.datasets.dataset import (
    Dataset,
    DatasetVersion,
    _batch_examples,
    example_from_dataset_entry,
    example_to_dataset_entry,
)


def _make_dataset(examples=None):
    return Dataset(
        name="test-ds",
        project_id="proj-1",
        project_name="my-project",
        examples=examples or [],
        client=None,
    )


class TestDatasetBasics:
    def test_len_empty(self):
        assert len(_make_dataset()) == 0

    def test_len_with_examples(self):
        examples = [Example.create(i=i) for i in range(5)]
        assert len(_make_dataset(examples)) == 5

    def test_iter(self):
        examples = [Example.create(i=i) for i in range(3)]
        ds = _make_dataset(examples)
        assert list(ds) == examples

    def test_str(self):
        ds = _make_dataset([Example()])
        assert "test-ds" in str(ds)
        assert "1" in str(ds)

    def test_iter_empty(self):
        ds = _make_dataset()
        assert list(ds) == []


class TestDatasetSave:
    def test_save_as_json(self):
        examples = [Example.create(input="q", output="a")]
        ds = _make_dataset(examples)
        with tempfile.TemporaryDirectory() as d:
            ds.save_as("json", d, save_name="out")
            path = os.path.join(d, "out.json")
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "examples" in data
            assert data["examples"][0]["input"] == "q"

    def test_save_as_yaml(self):
        import yaml

        examples = [Example.create(label="test")]
        ds = _make_dataset(examples)
        with tempfile.TemporaryDirectory() as d:
            ds.save_as("yaml", d, save_name="out")
            path = os.path.join(d, "out.yaml")
            assert os.path.exists(path)
            with open(path) as f:
                data = yaml.safe_load(f)
            assert "examples" in data

    def test_save_creates_dir(self):
        ds = _make_dataset()
        with tempfile.TemporaryDirectory() as base:
            subdir = os.path.join(base, "subdir")
            ds.save_as("json", subdir, save_name="file")
            assert os.path.exists(os.path.join(subdir, "file.json"))

    def test_save_no_examples_writes_empty(self):
        ds = _make_dataset()
        with tempfile.TemporaryDirectory() as d:
            ds.save_as("json", d, save_name="empty")
            with open(os.path.join(d, "empty.json")) as f:
                data = json.load(f)
            assert data["examples"] == []


class TestAddExamples:
    def test_add_examples_no_client_noop(self):
        ds = _make_dataset()
        ds.add_examples([Example.create(x=1)])

    def test_add_examples_single_raises_type_error(self):
        client = MagicMock()
        ds = _make_dataset()
        ds.client = client
        with pytest.raises(TypeError):
            ds.add_examples(Example.create(x=1))

    def test_add_examples_calls_client(self):
        client = MagicMock()
        client.post_projects_datasets_by_dataset_identifier_examples.return_value = {
            "example_ids": [],
            "version_added": 2,
        }
        ds = Dataset(
            name="test-ds",
            project_id="proj-1",
            project_name="proj",
            examples=[],
            client=client,
        )
        examples = [Example.create(x=i) for i in range(3)]
        ds.add_examples(examples, batch_size=2)
        assert (
            client.post_projects_datasets_by_dataset_identifier_examples.call_count == 2
        )
        assert ds.current_version == 2

    def test_add_examples_prefers_dataset_id_identifier(self):
        client = MagicMock()
        client.post_projects_datasets_by_dataset_identifier_examples.return_value = {}
        ds = Dataset(
            name="test-ds",
            project_id="proj-1",
            project_name="proj",
            dataset_id="ds-uuid",
            examples=[],
            client=client,
        )
        ds.add_examples([Example.create(x=1)])
        call = client.post_projects_datasets_by_dataset_identifier_examples.call_args
        assert call.kwargs["dataset_identifier"] == "ds-uuid"

    def test_add_examples_maps_validation_error(self):
        from judgeval.exceptions import JudgmentAPIError, JudgmentValidationError

        client = MagicMock()
        client.post_projects_datasets_by_dataset_identifier_examples.side_effect = (
            JudgmentAPIError(422, "examples failed schema validation", None)
        )
        ds = Dataset(
            name="test-ds",
            project_id="proj-1",
            project_name="proj",
            examples=[],
            client=client,
        )
        with pytest.raises(JudgmentValidationError):
            ds.add_examples([Example.create(x=1)])


class TestVersionsAndDelete:
    def test_versions_returns_dataset_versions(self):
        client = MagicMock()
        client.get_projects_datasets_by_dataset_identifier_versions.return_value = {
            "versions": [
                {
                    "version_id": "v2",
                    "dataset_id": "d1",
                    "version_number": 2,
                    "item_count": 5,
                },
                {
                    "version_id": "v1",
                    "dataset_id": "d1",
                    "version_number": 1,
                    "item_count": 3,
                },
            ]
        }
        ds = Dataset(
            name="test-ds",
            project_id="proj-1",
            project_name="proj",
            client=client,
        )
        versions = ds.versions()
        assert len(versions) == 2
        assert isinstance(versions[0], DatasetVersion)
        assert versions[0].version_number == 2
        assert versions[1].item_count == 3

    def test_delete_calls_client(self):
        client = MagicMock()
        ds = Dataset(
            name="test-ds",
            project_id="proj-1",
            project_name="proj",
            dataset_id="ds-uuid",
            client=client,
        )
        ds.delete()
        call = client.delete_projects_datasets_by_dataset_identifier.call_args
        assert call.kwargs["dataset_identifier"] == "ds-uuid"


class TestDatasetEntrySerialization:
    def test_example_to_dataset_entry_lifts_reserved_keys(self):
        example = Example.create(input="q")
        entry = example_to_dataset_entry(example)
        assert entry["example_id"] == example.example_id
        assert entry["created_at"] == example.created_at
        assert entry["input"] == "q"
        assert "name" not in entry

    def test_example_to_dataset_entry_passes_trace_column_string(self):
        example = Example.create(question="q", transcript="trace-1")
        entry = example_to_dataset_entry(example)
        assert entry["transcript"] == "trace-1"
        assert entry["question"] == "q"

    def test_example_from_dataset_entry_nested_data(self):
        entry = {
            "example_id": "e1",
            "created_at": "2026-01-01",
            "data": {"input": "q", "expected_output": "a"},
            "offline_trace_id": "trace-1",
        }
        example = example_from_dataset_entry(entry)
        assert example.example_id == "e1"
        assert example["input"] == "q"
        assert example["expected_output"] == "a"
        assert example["offline_trace_id"] == "trace-1"

    def test_example_from_dataset_entry_flat_fallback(self):
        entry = {"example_id": "e1", "created_at": "2026-01-01", "input": "q"}
        example = example_from_dataset_entry(entry)
        assert example["input"] == "q"


class TestBatchExamples:
    def test_exact_batches(self):
        examples = [Example() for _ in range(6)]
        batches = list(_batch_examples(examples, batch_size=2))
        assert len(batches) == 3
        assert all(len(b) == 2 for b in batches)

    def test_partial_last_batch(self):
        examples = [Example() for _ in range(5)]
        batches = list(_batch_examples(examples, batch_size=2))
        assert len(batches) == 3
        assert len(batches[-1]) == 1

    def test_empty(self):
        assert list(_batch_examples([], batch_size=10)) == []
