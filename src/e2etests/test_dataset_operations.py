"""
Tests for dataset operations in the JudgmentClient.
"""

import pytest
import json
import random
import string

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example


@pytest.mark.basic
class TestDatasetOperations:
    def test_dataset(self, client: JudgmentClient, project_name: str):
        """Test dataset creation and manipulation."""
        dataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))

        client.push_dataset(
            alias="test_dataset_5",
            dataset=dataset,
            project_name=project_name,
            overwrite=False,
        )

        dataset = client.pull_dataset(alias="test_dataset_5", project_name=project_name)
        assert dataset, "Failed to pull dataset"

        client.delete_dataset(alias="test_dataset_5", project_name=project_name)

    def test_pull_all_project_dataset_stats(
        self, client: JudgmentClient, project_name: str
    ):
        """Test pulling statistics for all project datasets."""
        dataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))
        dataset.add_example(Example(input="input 2", actual_output="output 2"))
        dataset.add_example(Example(input="input 3", actual_output="output 3"))
        random_name1 = "".join(
            random.choices(string.ascii_letters + string.digits, k=20)
        )
        client.push_dataset(
            alias=random_name1,
            dataset=dataset,
            project_name=project_name,
            overwrite=False,
        )

        dataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))
        dataset.add_example(Example(input="input 2", actual_output="output 2"))
        random_name2 = "".join(
            random.choices(string.ascii_letters + string.digits, k=20)
        )
        client.push_dataset(
            alias=random_name2,
            dataset=dataset,
            project_name=project_name,
            overwrite=False,
        )

        all_datasets_stats = client.pull_project_dataset_stats(project_name)

        assert all_datasets_stats, "Failed to pull dataset"
        assert all_datasets_stats[random_name1]["example_count"] == 3, (
            f"{random_name1} should have 3 examples"
        )
        assert all_datasets_stats[random_name2]["example_count"] == 2, (
            f"{random_name2} should have 2 examples"
        )

        client.delete_dataset(alias=random_name1, project_name=project_name)
        client.delete_dataset(alias=random_name2, project_name=project_name)

    def test_append_dataset(self, client: JudgmentClient, project_name: str):
        """Test dataset editing."""
        dataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))
        dataset.add_example(Example(input="input 2", actual_output="output 2"))
        client.push_dataset(
            alias="test_dataset_6",
            dataset=dataset,
            project_name=project_name,
            overwrite=True,
        )
        dataset = client.pull_dataset(
            alias="test_dataset_6", project_name=project_name
        )  # Pull in case dataset already has examples

        initial_example_count = len(dataset.examples)
        assert initial_example_count == 2, "Dataset should have 2 examples"

        client.append_dataset(
            alias="test_dataset_6",
            examples=[Example(input="input 3", actual_output="output 3")],
            project_name=project_name,
        )
        dataset = client.pull_dataset(alias="test_dataset_6", project_name=project_name)
        assert dataset, "Failed to pull dataset"
        assert len(dataset.examples) == initial_example_count + 1, (
            f"Dataset should have {initial_example_count + 1} examples, but has {len(dataset.examples)}"
        )

        client.delete_dataset(alias="test_dataset_6", project_name=project_name)

    def test_overwrite_dataset(self, client: JudgmentClient, project_name: str):
        """Test dataset overwriting."""
        dataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))
        client.push_dataset(
            alias="test_dataset_7",
            dataset=dataset,
            project_name=project_name,
            overwrite=True,
        )

        dataset = client.create_dataset()
        dataset.add_example(Example(input="input 2", actual_output="output 2"))
        dataset.add_example(Example(input="input 3", actual_output="output 3"))
        client.push_dataset(
            alias="test_dataset_7",
            dataset=dataset,
            project_name=project_name,
            overwrite=True,
        )

        dataset = client.pull_dataset(alias="test_dataset_7", project_name=project_name)
        assert dataset, "Failed to pull dataset"
        assert len(dataset.examples) == 2, "Dataset should have 2 examples"

    def test_append_dataset2(self, client: JudgmentClient, project_name: str):
        """Test dataset appending."""
        dataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))
        client.push_dataset(
            alias="test_dataset_8",
            dataset=dataset,
            project_name=project_name,
            overwrite=True,
        )

        examples = [
            Example(input="input 2", actual_output="output 2"),
            Example(input="input 3", actual_output="output 3"),
        ]
        client.append_dataset(
            alias="test_dataset_8", examples=examples, project_name=project_name
        )

        dataset = client.pull_dataset(alias="test_dataset_8", project_name=project_name)
        assert dataset, "Failed to pull dataset"
        assert len(dataset.examples) == 3, "Dataset should have 3 examples"

    def test_export_jsonl(
        self, client: JudgmentClient, random_name: str, project_name: str
    ):
        """Test JSONL dataset export functionality."""
        # Create and push test dataset
        dataset = client.create_dataset()
        dataset.add_example(
            Example(
                input="Test input 1",
                actual_output="Test output 1",
                expected_output="Expected output 1",
            )
        )
        client.push_dataset(
            alias=random_name,
            dataset=dataset,
            project_name=project_name,
            overwrite=True,
        )

        # Export as JSONL
        response = client.eval_dataset_client.export_jsonl(random_name, project_name)
        assert response.status_code == 200, "Export request failed"

        # Validate JSONL format and content
        example_count = 0

        for line in response.iter_lines():
            if line:
                entry = json.loads(line.decode("utf-8"))
                assert "input" in entry, "Missing input field"
                assert "output" in entry, "Missing output field"
                assert "source" in entry, "Missing source field"

                if entry["source"] == "example":
                    example_count += 1
                    assert "expected_output" in entry, "Example missing expected_output"

        assert example_count == 1, f"Expected 1 example, got {example_count}"

        client.delete_dataset(alias=random_name, project_name=project_name)
