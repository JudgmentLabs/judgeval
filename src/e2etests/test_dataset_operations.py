import random
import string
import pytest
from judgeval.v1 import Judgeval
from judgeval.exceptions import JudgmentAPIError
from judgeval.v1.data import Example
from judgeval.v1.datasets.dataset import DatasetInfo
from e2etests.utils import create_project, delete_project


def test_create_dataset(client: Judgeval, random_name: str):
    client.datasets.create(name=random_name)


def test_create_dataset_with_example(client: Judgeval, random_name: str):
    dataset = client.datasets.create(
        name=random_name,
        examples=[Example.create(input="input 1", actual_output="output 1")],
    )
    assert dataset, "Failed to push dataset"


def test_create_dataset_across_projects(client: Judgeval, random_name: str):
    create_project(project_name=random_name)
    dataset = client.datasets.create(
        name=random_name,
        examples=[Example.create(input="input 1", actual_output="output 1")],
    )
    assert dataset, "Failed to push dataset"

    client2 = Judgeval(project_name=random_name)
    dataset2 = client2.datasets.create(
        name=random_name,
        examples=[Example.create(input="input 1", actual_output="output 1")],
    )
    assert dataset2, "Failed to push dataset"
    delete_project(project_name=random_name)


def test_create_dataset_error(client: Judgeval, random_name: str):
    dataset = client.datasets.create(
        name=random_name,
        examples=[Example.create(input="input 1", actual_output="output 1")],
    )
    assert dataset

    with pytest.raises(JudgmentAPIError):
        client.datasets.create(
            name=random_name,
            examples=[Example.create(input="input 1", actual_output="output 1")],
        )


def test_get_dataset_error(client: Judgeval, random_name: str):
    with pytest.raises(JudgmentAPIError):
        client.datasets.get(name=random_name)


def test_pull_dataset(client: Judgeval):
    examples = [
        Example.create(input="input 1", actual_output="output 1"),
        Example.create(input="input 2", actual_output="output 2"),
        Example.create(input="input 3", actual_output="output 3"),
        Example.create(input="input 4", actual_output="output 4"),
        Example.create(input="input 5", actual_output="output 5"),
    ]
    random_name1 = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    client.datasets.create(name=random_name1, examples=examples[:3])

    random_name2 = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    client.datasets.create(name=random_name2, examples=examples[3:])

    dataset1 = client.datasets.get(name=random_name1)
    dataset2 = client.datasets.get(name=random_name2)

    assert dataset1, "Failed to pull dataset"
    assert dataset1.name == random_name1
    assert len(dataset1) == 3, "Dataset should have 3 examples"
    for i, e in enumerate(dataset1, start=1):
        assert e.get_property("input") == f"input {i}"
        assert e.get_property("actual_output") == f"output {i}"

    assert dataset2, "Failed to pull dataset"
    assert dataset2.name == random_name2
    assert len(dataset2) == 2, "Dataset should have 2 examples"
    for i, e in enumerate(dataset2, start=4):
        assert e.get_property("input") == f"input {i}"
        assert e.get_property("actual_output") == f"output {i}"


def test_append_dataset(client: Judgeval, random_name: str):
    examples = [
        Example.create(input="input 1", actual_output="output 1"),
        Example.create(input="input 2", actual_output="output 2"),
    ]
    client.datasets.create(name=random_name, examples=examples)
    dataset = client.datasets.get(name=random_name)

    initial_example_count = len(dataset)
    new_examples = [
        Example.create(input="input 3", actual_output="output 3"),
        Example.create(input="input 4", actual_output="output 4"),
        Example.create(input="input 5", actual_output="output 5"),
    ]
    assert initial_example_count == 2, "Dataset should have 2 examples"
    dataset.add_examples(new_examples)

    dataset = client.datasets.get(name=random_name)
    assert dataset, "Failed to pull dataset"
    assert len(dataset) == initial_example_count + 3
    for i, e in enumerate(dataset, start=1):
        assert e.get_property("input") == f"input {i}"
        assert e.get_property("actual_output") == f"output {i}"


def test_add_examples_error(client: Judgeval, random_name: str):
    dataset = client.datasets.create(name=random_name)
    with pytest.raises(TypeError):
        ex = Example.create(input="input 1", actual_output="output 1")
        dataset.add_examples(ex)


def test_overwrite_dataset(client: Judgeval, random_name: str):
    examples = [
        Example.create(input="input 1", actual_output="output 1"),
        Example.create(input="input 2", actual_output="output 2"),
    ]
    client.datasets.create(name=random_name, examples=examples)

    new_examples = [
        Example.create(input="input 3", actual_output="output 3"),
        Example.create(input="input 4", actual_output="output 4"),
    ]
    client.datasets.create(
        name=random_name,
        examples=new_examples,
        overwrite=True,
    )
    dataset = client.datasets.get(name=random_name)
    assert dataset, "Failed to pull dataset"
    assert len(dataset) == 2, "Dataset should have 2 examples"


def test_dataset_list_empty(client: Judgeval, random_name: str):
    create_project(project_name=random_name)

    try:
        client2 = Judgeval(project_name=random_name)
        datasets = client2.datasets.list()
        assert datasets == [], "Empty project should return empty list of datasets"
    finally:
        delete_project(project_name=random_name)


def test_dataset_list_nonexistent_project(client: Judgeval, random_name: str):
    client2 = Judgeval(project_name=random_name)
    datasets = client2.datasets.list()
    assert datasets is None or datasets == []


def test_dataset_list_after_creation(client: Judgeval, random_name: str):
    initial_datasets = client.datasets.list()
    initial_count = len(initial_datasets)

    dataset_name1 = random_name + "_1"
    dataset_name2 = random_name + "_2"

    examples1 = [Example.create(input="input 1", actual_output="output 1")]
    examples2 = [Example.create(input="input 2", actual_output="output 2")]

    client.datasets.create(name=dataset_name1, examples=examples1)
    client.datasets.create(name=dataset_name2, examples=examples2)

    updated_datasets = client.datasets.list()

    assert len(updated_datasets) == initial_count + 2

    dataset_names = [d.name for d in updated_datasets]
    assert dataset_name1 in dataset_names
    assert dataset_name2 in dataset_names

    for dataset_info in updated_datasets:
        assert isinstance(dataset_info, DatasetInfo)
        assert dataset_info.kind == "example"
        assert dataset_info.entries >= 0


def test_dataset_list_reflects_changes(client: Judgeval, random_name: str):
    examples = [
        Example.create(input="input 1", actual_output="output 1"),
        Example.create(input="input 2", actual_output="output 2"),
    ]
    client.datasets.create(name=random_name, examples=examples)

    datasets = client.datasets.list()
    dataset_names = [d.name for d in datasets]
    assert random_name in dataset_names

    created_dataset_info = next(d for d in datasets if d.name == random_name)
    assert created_dataset_info.entries == 2

    client.datasets.create(
        name=random_name,
        examples=[],
        overwrite=True,
    )

    updated_datasets = client.datasets.list()
    updated_dataset_names = [d.name for d in updated_datasets]
    assert random_name in updated_dataset_names

    overwritten_dataset_info = next(
        d for d in updated_datasets if d.name == random_name
    )
    assert overwritten_dataset_info.entries == 0
