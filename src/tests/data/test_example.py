"""
Unit tests for the Example class
"""

import pytest
from pydantic import ValidationError
from judgeval.data import Example


def test_basic_example_creation():
    example = Example(input="test input", actual_output="test output")
    assert example.input == "test input"
    assert example.actual_output == "test output"
    assert example.expected_output is None
    assert example.created_at is not None
    # Verify timestamp format


def test_to_dict():
    example = Example(
        input="test input", actual_output="test output", name="test example"
    )

    example_dict = example.to_dict()
    assert example_dict["input"] == "test input"
    assert example_dict["actual_output"] == "test output"
    assert example_dict["name"] == "test example"
    assert "created_at" in example_dict


def test_string_representation():
    example = Example(input="test input", actual_output="test output")

    str_repr = str(example)
    assert "input=test input" in str_repr
    assert "actual_output=test output" in str_repr


# Error cases
def test_invalid_context_type():
    with pytest.raises(ValidationError):
        Example(
            input="test",
            actual_output="test",
            context="invalid context type",  # Should be list of strings
        )


def test_invalid_context_content():
    with pytest.raises(ValidationError):
        Example(
            input="test",
            actual_output="test",
            context=["valid", 123],  # Should be all strings
        )


def test_invalid_retrieval_context():
    with pytest.raises(ValidationError):
        Example(
            input="test",
            actual_output="test",
            retrieval_context=[1, 2, 3],  # Should be list of strings
        )


def test_invalid_tools_called():
    with pytest.raises(ValidationError):
        Example(
            input="test",
            actual_output="test",
            tools_called={"tool1": "value"},  # Should be list of strings
        )


def test_invalid_expected_tools():
    with pytest.raises(ValidationError):
        Example(
            input="test",
            actual_output="test",
            expected_tools=[1, "tool2"],  # Should be list of strings
        )
