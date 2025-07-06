import yaml
import json
from typing import List
from judgeval.common.logger import debug, info, error

from judgeval.data import Example

"""
File utilities for loading test examples from different file formats.

Usage Examples:
    # Load from YAML file
    from judgeval.utils.file_utils import get_examples_from_yaml
    examples = get_examples_from_yaml("tests.yaml")

    # Load from JSON file
    from judgeval.utils.file_utils import get_examples_from_json
    examples = get_examples_from_json("tests.json")

    # Use in evaluation
    client.assert_test(
        examples=examples,
        scorers=[FaithfulnessScorer(threshold=0.5)],
        model="gpt-4.1",
    )
"""


def get_examples_from_yaml(file_path: str) -> List[Example] | None:
    debug(f"Loading dataset from YAML file: {file_path}")
    """
    Adds examples from a YAML file.

    The format of the YAML file is expected to be a dictionary with one key: "examples".
    The value of the key is a list of dictionaries, where each dictionary represents an example.

    The YAML file is expected to have the following format:
    examples:
        - input: "test input"
        actual_output: "test output"
        expected_output: "expected output"
        context:
            - "context1"
            - "context2"
        retrieval_context:
            - "retrieval1"
        additional_metadata:
            key: "value"
        tools_called:
            - "tool1"
        expected_tools:
            - {tool_name: "tool1", parameters: {"query": "test query 1"}}
            - {tool_name: "tool2", parameters: {"query": "test query 2"}}
        name: "test example"
        example_id: null
        timestamp: "20241230_160117"
        trace_id: "123"
    """
    try:
        with open(file_path, "r") as file:
            payload = yaml.safe_load(file)
            if payload is None:
                raise ValueError("The YAML file is empty.")
            examples = payload.get("examples", [])
    except FileNotFoundError:
        error(f"YAML file not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except yaml.YAMLError:
        error(f"Invalid YAML file: {file_path}")
        raise ValueError(f"The file {file_path} is not a valid YAML file.")

    info(f"Added {len(examples)} examples from YAML")
    new_examples = [Example(**e) for e in examples]
    return new_examples


def get_examples_from_json(file_path: str) -> List[Example] | None:
    debug(f"Loading dataset from JSON file: {file_path}")
    """
    Adds examples from a JSON file.

    The format of the JSON file is expected to be a dictionary with one key: "examples".
    The value of the key is a list of dictionaries, where each dictionary represents an example.

    The JSON file is expected to have the following format:
    {
        "examples": [
            {
                "input": "test input",
                "actual_output": "test output",
                "expected_output": "expected output",
                "context": [
                    "context1",
                    "context2"
                ],
                "retrieval_context": [
                    "retrieval1"
                ],
                "additional_metadata": {
                    "key": "value"
                },
                "tools_called": [
                    "tool1"
                ],
                "expected_tools": [
                    {"tool_name": "tool1", "parameters": {"query": "test query 1"}},
                    {"tool_name": "tool2", "parameters": {"query": "test query 2"}}
                ],
                "name": "test example",
                "example_id": null,
                "timestamp": "20241230_160117",
                "trace_id": "123"
            }
        ]
    }
    """
    try:
        with open(file_path, "r") as file:
            payload = json.load(file)
            if payload is None:
                raise ValueError("The JSON file is empty.")
            examples = payload.get("examples", [])
    except FileNotFoundError:
        error(f"JSON file not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        error(f"Invalid JSON file: {file_path} - {str(e)}")
        raise ValueError(f"The file {file_path} is not a valid JSON file: {str(e)}")

    info(f"Added {len(examples)} examples from JSON")
    new_examples = [Example(**e) for e in examples]
    return new_examples
