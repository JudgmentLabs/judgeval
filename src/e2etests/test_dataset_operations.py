"""
Tests for dataset operations in the JudgmentClient.
"""

import random
import string
import pytest
from judgeval import JudgmentClient
from judgeval.exceptions import JudgmentAPIError
from judgeval.data import Example
from judgeval.dataset import Dataset
from e2etests.utils import create_project, delete_project


def test_create_dataset(client: JudgmentClient, project_name: str, random_name: str):
    """Test dataset creation"""
    Dataset.create(
        name=random_name,
        project_name=project_name,
    )


def test_create_dataset_with_example(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test dataset creation and manipulation."""
    dataset = Dataset.create(
        name=random_name,
        project_name=project_name,
        examples=[Example(input="input 1", actual_output="output 1")],
    )
    assert dataset, "Failed to push dataset"


def test_create_dataset_across_projects(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test that the same name for a dataset can be used across projects."""
    create_project(project_name=random_name)
    dataset = Dataset.create(
        name=random_name,
        project_name=project_name,
        examples=[Example(input="input 1", actual_output="output 1")],
    )

    assert dataset, "Failed to push dataset"

    dataset2 = Dataset.create(
        name=random_name,
        project_name=random_name,
        examples=[Example(input="input 1", actual_output="output 1")],
    )

    assert dataset2, "Failed to push dataset"
    delete_project(project_name=random_name)


def test_create_dataset_error(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test that the same name for a dataset can be used across projects."""
    dataset = Dataset.create(
        name=random_name,
        project_name=project_name,
        examples=[Example(input="input 1", actual_output="output 1")],
    )
    assert dataset

    try:
        Dataset.create(
            name=random_name,
            project_name=project_name,
            examples=[Example(input="input 1", actual_output="output 1")],
        )
    except Exception as e:
        assert "Dataset already exists" in str(e)


def test_get_dataset_error(client: JudgmentClient, project_name: str, random_name: str):
    """Test that the dataset is not found."""
    with pytest.raises(JudgmentAPIError):
        Dataset.get(name=random_name, project_name=project_name)


def test_pull_dataset(client: JudgmentClient, project_name: str):
    """Test pulling statistics for all project datasets."""
    examples = [
        Example(input="input 1", actual_output="output 1"),
        Example(input="input 2", actual_output="output 2"),
        Example(input="input 3", actual_output="output 3"),
        Example(input="input 4", actual_output="output 4"),
        Example(input="input 5", actual_output="output 5"),
    ]
    random_name1 = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    Dataset.create(name=random_name1, project_name=project_name, examples=examples[:3])

    random_name2 = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    Dataset.create(name=random_name2, project_name=project_name, examples=examples[3:])

    dataset1 = Dataset.get(name=random_name1, project_name=project_name)
    dataset2 = Dataset.get(name=random_name2, project_name=project_name)

    assert dataset1, "Failed to pull dataset"
    assert dataset1.name == random_name1, (
        "Dataset name should be the same as the one used to create it"
    )
    assert len(dataset1.examples) == 3, "Dataset should have 3 examples"
    for i, e in enumerate(dataset1.examples, start=1):
        assert e.input == f"input {i}", (
            f"Example should have .input be 'input {i}' but got '{e.input}'"
        )
        assert e.actual_output == f"output {i}", (
            f"Example should have .actual_output be 'output {i}' but got '{e.actual_output}'"
        )

    assert dataset2, "Failed to pull dataset"
    assert dataset2.name == random_name2, (
        "Dataset name should be the same as the one used to create it"
    )
    assert len(dataset2.examples) == 2, "Dataset should have 2 examples"
    for i, e in enumerate(dataset2.examples, start=4):
        assert e.input == f"input {i}", (
            f"Example should have .input be 'input {i}' but got '{e.input}'"
        )
        assert e.actual_output == f"output {i}", (
            f"Example should have .actual_output be 'output {i}' but got '{e.actual_output}'"
        )


def test_append_dataset(client: JudgmentClient, project_name: str, random_name: str):
    """Test dataset editing."""
    examples = [
        Example(input="input 1", actual_output="output 1"),
        Example(input="input 2", actual_output="output 2"),
    ]
    Dataset.create(name=random_name, project_name=project_name, examples=examples)
    dataset = Dataset.get(name=random_name, project_name=project_name)

    initial_example_count = len(dataset.examples)
    examples = [
        Example(input="input 3", actual_output="output 3"),
        Example(input="input 4", actual_output="output 4"),
        Example(input="input 5", actual_output="output 5"),
    ]
    assert initial_example_count == 2, "Dataset should have 2 examples"
    dataset.add_examples(examples)

    dataset = Dataset.get(name=random_name, project_name=project_name)
    assert dataset, "Failed to pull dataset"
    assert len(dataset.examples) == initial_example_count + 3, (
        f"Dataset should have {initial_example_count + 3} examples, but has {len(dataset.examples)}"
    )


def test_overwrite_dataset(client: JudgmentClient, project_name: str, random_name: str):
    """Test dataset overwriting."""
    examples = [
        Example(input="input 1", actual_output="output 1"),
        Example(input="input 2", actual_output="output 2"),
    ]
    Dataset.create(name=random_name, project_name=project_name, examples=examples)
    dataset = Dataset.get(name=random_name, project_name=project_name)

    new_examples = [
        Example(input="input 3", actual_output="output 3"),
        Example(input="input 4", actual_output="output 4"),
    ]
    Dataset.create(
        name=random_name,
        project_name=project_name,
        examples=new_examples,
        overwrite=True,
    )
    dataset = Dataset.get(name=random_name, project_name=project_name)
    assert dataset, "Failed to pull dataset"
    assert len(dataset.examples) == 2, "Dataset should have 2 examples"
