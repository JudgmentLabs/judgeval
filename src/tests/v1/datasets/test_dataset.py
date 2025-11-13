import pytest
import tempfile
import os
import orjson
import yaml
from unittest.mock import MagicMock
from judgeval.v1.datasets.dataset import Dataset, DatasetInfo
from judgeval.v1.datasets.dataset_factory import DatasetFactory
from judgeval.v1.data.example import Example


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def factory(mock_client):
    return DatasetFactory(mock_client)


@pytest.fixture
def sample_examples():
    return [
        Example(name="example1")
        .set_property("input", "input1")
        .set_property("output", "output1"),
        Example(name="example2")
        .set_property("input", "input2")
        .set_property("output", "output2"),
        Example(name="example3")
        .set_property("input", "input3")
        .set_property("output", "output3"),
    ]


def test_dataset_get(factory, mock_client, sample_examples):
    mock_client.datasets_pull_for_judgeval.return_value = {
        "dataset_kind": "example",
        "examples": [
            {
                "data": {
                    "example_id": "1",
                    "created_at": "2024-01-01",
                    "name": "example1",
                    "input": "input1",
                    "output": "output1",
                }
            }
        ],
    }

    dataset = factory.get("test_dataset", "test_project")

    assert dataset.name == "test_dataset"
    assert dataset.project_name == "test_project"
    assert dataset.dataset_kind == "example"
    assert len(dataset.examples) == 1
    assert dataset.examples[0].get_property("input") == "input1"
    assert dataset.examples[0].get_property("output") == "output1"
    mock_client.datasets_pull_for_judgeval.assert_called_once()


def test_dataset_create(factory, mock_client, sample_examples):
    dataset = factory.create(
        name="test_dataset",
        project_name="test_project",
        examples=sample_examples,
        overwrite=False,
    )

    assert dataset.name == "test_dataset"
    assert dataset.project_name == "test_project"
    assert len(dataset.examples) == 3
    mock_client.datasets_create_for_judgeval.assert_called_once()


def test_dataset_list(factory, mock_client):
    mock_client.datasets_pull_all_for_judgeval.return_value = [
        {
            "dataset_id": "1",
            "name": "dataset1",
            "created_at": "2024-01-01",
            "kind": "example",
            "entries": 10,
            "creator": "user1",
        },
        {
            "dataset_id": "2",
            "name": "dataset2",
            "created_at": "2024-01-02",
            "kind": "example",
            "entries": 20,
            "creator": "user2",
        },
    ]

    datasets = factory.list("test_project")

    assert len(datasets) == 2
    assert isinstance(datasets[0], DatasetInfo)
    assert datasets[0].name == "dataset1"
    assert datasets[0].entries == 10
    assert datasets[1].name == "dataset2"
    assert datasets[1].entries == 20


def test_dataset_add_examples(mock_client, sample_examples):
    dataset = Dataset(
        name="test_dataset",
        project_name="test_project",
        examples=sample_examples,
        client=mock_client,
    )

    new_examples = [
        Example(name="example4").set_property("input", "input4"),
    ]

    dataset.add_examples(new_examples)

    mock_client.datasets_insert_examples_for_judgeval.assert_called_once()


def test_dataset_iteration(sample_examples):
    dataset = Dataset(
        name="test_dataset", project_name="test_project", examples=sample_examples
    )

    count = 0
    for example in dataset:
        count += 1
        assert isinstance(example, Example)

    assert count == 3


def test_dataset_length(sample_examples):
    dataset = Dataset(
        name="test_dataset", project_name="test_project", examples=sample_examples
    )

    assert len(dataset) == 3


def test_dataset_str(sample_examples):
    dataset = Dataset(
        name="test_dataset", project_name="test_project", examples=sample_examples
    )

    str_repr = str(dataset)
    assert "test_dataset" in str_repr
    assert "3" in str_repr


def test_dataset_save_as_json(sample_examples):
    dataset = Dataset(
        name="test_dataset", project_name="test_project", examples=sample_examples
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset.save_as("json", tmpdir, "test_output")

        output_path = os.path.join(tmpdir, "test_output.json")
        assert os.path.exists(output_path)

        with open(output_path, "rb") as f:
            data = orjson.loads(f.read())

        assert "examples" in data
        assert len(data["examples"]) == 3


def test_dataset_save_as_yaml(sample_examples):
    dataset = Dataset(
        name="test_dataset", project_name="test_project", examples=sample_examples
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset.save_as("yaml", tmpdir, "test_output")

        output_path = os.path.join(tmpdir, "test_output.yaml")
        assert os.path.exists(output_path)

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        assert "examples" in data
        assert len(data["examples"]) == 3


def test_dataset_add_from_json(mock_client):
    dataset = Dataset(
        name="test_dataset",
        project_name="test_project",
        examples=[],
        client=mock_client,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test.json")
        test_data = [
            {"name": "example1", "input": "input1", "output": "output1"},
            {"name": "example2", "input": "input2", "output": "output2"},
        ]
        with open(json_path, "wb") as f:
            f.write(orjson.dumps(test_data))

        dataset.add_from_json(json_path)

        mock_client.datasets_insert_examples_for_judgeval.assert_called_once()


def test_dataset_add_from_yaml(mock_client):
    dataset = Dataset(
        name="test_dataset",
        project_name="test_project",
        examples=[],
        client=mock_client,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "test.yaml")
        test_data = [
            {"name": "example1", "input": "input1", "output": "output1"},
            {"name": "example2", "input": "input2", "output": "output2"},
        ]
        with open(yaml_path, "w") as f:
            yaml.dump(test_data, f)

        dataset.add_from_yaml(yaml_path)

        mock_client.datasets_insert_examples_for_judgeval.assert_called_once()


def test_dataset_display(sample_examples, capsys):
    dataset = Dataset(
        name="test_dataset", project_name="test_project", examples=sample_examples
    )

    dataset.display(max_examples=2)

    captured = capsys.readouterr()
    assert "test_dataset" in captured.out
    assert "test_project" in captured.out


def test_dataset_empty():
    dataset = Dataset(name="test_dataset", project_name="test_project", examples=[])

    assert len(dataset) == 0
    count = 0
    for _ in dataset:
        count += 1
    assert count == 0
