import yaml
from typing import List
from judgeval.common.logger import judgeval_logger

from judgeval.data import Example


def get_examples_from_yaml(file_path: str) -> List[Example] | None:
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
        judgeval_logger.error(f"YAML file not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except yaml.YAMLError:
        judgeval_logger.error(f"Invalid YAML file: {file_path}")
        raise ValueError(f"The file {file_path} is not a valid YAML file.")

    new_examples = [Example(**e) for e in examples]
    return new_examples
