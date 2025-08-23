import importlib.util
import yaml
import orjson
from pathlib import Path
from typing import List
from judgeval.logger import judgeval_logger

from judgeval.data.example import Example


def get_examples_from_yaml(file_path: str) -> List[Example]:
    """
    Adds examples from a YAML file.

    The YAML file is expected to have the following format:
    - key_01: value_01
        key_02: value_02
    - key_11: value_11
        key_12: value_12
        key_13: value_13
    ...
    """
    try:
        with open(file_path, "r") as file:
            payload = yaml.safe_load(file)
            if payload is None:
                raise ValueError("The YAML file is empty.")
    except FileNotFoundError:
        judgeval_logger.error(f"YAML file not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except yaml.YAMLError:
        judgeval_logger.error(f"Invalid YAML file: {file_path}")
        raise ValueError(f"The file {file_path} is not a valid YAML file.")

    new_examples = [Example(**e) for e in payload]
    return new_examples


def get_examples_from_json(file_path: str) -> List[Example]:
    """
    Adds examples from a JSON file.

    The JSON file is expected to have the following format:
    [
        {
            "key_01": "value_01",
            "key_02": "value_02"
        },
        {
            "key_11": "value_11",
            "key_12": "value_12",
            "key_13": "value_13"
        },
        ...
    ]
    """
    try:
        with open(file_path, "rb") as file:
            payload = orjson.loads(file.read())
    except FileNotFoundError:
        judgeval_logger.error(f"JSON file not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except orjson.JSONDecodeError:
        judgeval_logger.error(f"Invalid JSON file: {file_path}")
        raise ValueError(f"The file {file_path} is not a valid JSON file.")

    new_examples = [Example(**e) for e in payload]
    return new_examples


def extract_scorer_name(scorer_file_path: str) -> str:
    try:
        spec = importlib.util.spec_from_file_location("scorer_module", scorer_file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from {scorer_file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and any("Scorer" in str(base) for base in attr.__mro__)
                and attr.__module__ == "scorer_module"
            ):
                try:
                    # Instantiate the scorer and get its name
                    scorer_instance = attr()
                    if hasattr(scorer_instance, "name"):
                        return scorer_instance.name
                except Exception:
                    # Skip if instantiation fails
                    continue

        raise AttributeError("No scorer class found or could be instantiated")
    except Exception as e:
        judgeval_logger.warning(f"Could not extract scorer name: {e}")
        return Path(scorer_file_path).stem
