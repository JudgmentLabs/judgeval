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
    def test_dataset(self, client: JudgmentClient):
        """Test dataset creation and manipulation."""
        dataset = client.create_dataset()
        
        # Correct way to create an Example with dictionary values
        dataset.add_example(Example(
            input="input 1", 
            actual_output={"response": "output 1"}
        ))
        
        client.push_dataset(alias="test_dataset_5", dataset=dataset, overwrite=False)
        dataset = client.pull_dataset(alias="test_dataset_5")
        assert dataset, "Failed to pull dataset"

    def test_pull_all_user_dataset_stats(self, client: JudgmentClient):
        """Test pulling statistics for all user datasets."""
        dataset = client.create_dataset()
        dataset.add_example(Example(input={"input": "input 1"}, actual_output={"output": "output 1"}))
        dataset.add_example(Example(input={"input": "input 2"}, actual_output={"output": "output 2"}))
        dataset.add_example(Example(input={"input": "input 3"}, actual_output={"output": "output 3"}))
        random_name1 = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        client.push_dataset(alias=random_name1, dataset=dataset, overwrite=False)

        dataset = client.create_dataset()
        dataset.add_example(Example(input={"input": "input 1"}, actual_output={"output": "output 1"}))
        dataset.add_example(Example(input={"input": "input 2"}, actual_output={"output": "output 2"}))
        random_name2 = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        client.push_dataset(alias=random_name2, dataset=dataset, overwrite=False)
        
        all_datasets_stats = client.pull_all_user_dataset_stats()
        print(all_datasets_stats)
        assert all_datasets_stats, "Failed to pull dataset"
        assert all_datasets_stats[random_name1]["example_count"] == 3, f"{random_name1} should have 3 examples"
        assert all_datasets_stats[random_name2]["example_count"] == 2, f"{random_name2} should have 2 examples"

    def test_edit_dataset(self, client: JudgmentClient):
        """Test dataset editing."""
        dataset = client.create_dataset()
        dataset.add_example(Example(input={"input": "input 1"}, actual_output={"output": "output 1"}))
        dataset.add_example(Example(input={"input": "input 2"}, actual_output={"output": "output 2"}))
        client.push_dataset(alias="test_dataset_6", dataset=dataset, overwrite=True)
        dataset = client.pull_dataset(alias="test_dataset_6") # Pull in case dataset already has examples

        initial_example_count = len(dataset.examples)

        client.edit_dataset(
            alias="test_dataset_6",
            examples=[Example(input={"input": "input 3"}, actual_output={"output": "output 3"})],
        )
        dataset = client.pull_dataset(alias="test_dataset_6")
        assert dataset, "Failed to pull dataset"
        assert len(dataset.examples) == initial_example_count + 1, \
            f"Dataset should have {initial_example_count + 1} examples, but has {len(dataset.examples)}"

    def test_export_jsonl(self, client: JudgmentClient, random_name: str):
        """Test JSONL dataset export functionality."""
        # Create and push test dataset
        dataset = client.create_dataset()
        dataset.add_example(Example(
            input={"input": "Test input 1"}, 
            actual_output={"output": "Test output 1"},
            expected_output={"output": "Expected output 1"}
        ))
        client.push_dataset(alias=random_name, dataset=dataset, overwrite=True)

        # Export as JSONL
        response = client.eval_dataset_client.export_jsonl(random_name)
        assert response.status_code == 200, "Export request failed"

        # Validate JSONL format and content
        example_count = 0
        
        for line in response.iter_lines():
            if line:
                entry = json.loads(line.decode('utf-8'))
                assert "input" in entry, "Missing input field"
                assert "output" in entry, "Missing output field"
                assert "source" in entry, "Missing source field"
                
                if entry["source"] == "example":
                    example_count += 1
                    assert "expected_output" in entry, "Example missing expected_output"

        assert example_count == 1, f"Expected 1 example, got {example_count}"
         