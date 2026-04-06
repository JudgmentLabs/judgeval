from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from judgeval.data.example import Example
from judgeval.datasets.dataset import Dataset, _batch_examples


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
        ds = Dataset(
            name="test-ds",
            project_id="proj-1",
            project_name="proj",
            examples=[],
            client=client,
        )
        examples = [Example.create(x=i) for i in range(3)]
        ds.add_examples(examples, batch_size=2)
        assert client.post_projects_datasets_by_dataset_name_examples.call_count >= 1


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
