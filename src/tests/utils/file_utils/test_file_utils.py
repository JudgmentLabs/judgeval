from judgeval.utils.file_utils import get_examples_from_yaml, get_examples_from_json
import pytest


def test_get_examples_from_yaml():
    examples = get_examples_from_yaml("tests/utils/file_utils/example.yaml")
    assert len(examples) == 3

    assert examples[0].key_01 == "value_01"
    assert examples[0].key_02 == "value_02"

    assert examples[1].key_11 == "value_11"
    assert examples[1].key_12 == "value_12"
    assert examples[1].key_13 == "value_13"

    assert examples[2].key_21 == "value_21"
    assert examples[2].key_22 == "value_22"
    assert examples[2].key_23 == "value_23"
    assert examples[2].key_24 == "value_24"


def test_get_examples_from_yaml_with_empty_file():
    with pytest.raises(ValueError):
        get_examples_from_yaml("tests/utils/file_utils/example_empty.yaml")


def test_get_examples_from_non_existent_yaml():
    with pytest.raises(FileNotFoundError):
        get_examples_from_yaml("tests/utils/file_utils/non_existent_file.yaml")


def test_get_examples_from_json():
    examples = get_examples_from_json("tests/utils/file_utils/example.json")
    assert len(examples) == 3

    assert examples[0].key_01 == "value_01"
    assert examples[0].key_02 == "value_02"

    assert examples[1].key_11 == "value_11"
    assert examples[1].key_12 == "value_12"
    assert examples[1].key_13 == "value_13"

    assert examples[2].key_21 == "value_21"
    assert examples[2].key_22 == "value_22"
    assert examples[2].key_23 == "value_23"
    assert examples[2].key_24 == "value_24"


def test_get_examples_from_json_with_empty_file():
    with pytest.raises(ValueError):
        get_examples_from_json("tests/utils/file_utils/example_empty.json")


def test_get_examples_from_non_existent_json():
    with pytest.raises(FileNotFoundError):
        get_examples_from_json("tests/utils/file_utils/non_existent_file.json")
