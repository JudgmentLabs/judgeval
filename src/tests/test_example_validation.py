"""
Tests for Example class validation, specifically testing that invalid keys are rejected.
"""

import pytest
from pydantic import ValidationError

from judgeval.data import Example


class TestExampleValidation:
    """Test cases for Example class field validation."""

    def test_invalid_key_raises_validation_error(self):
        """Test that passing an invalid key raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Example(invalid_key='test', input='hello world')
        
        error = exc_info.value
        assert "Extra inputs are not permitted" in str(error)
        assert "invalid_key" in str(error)

    def test_multiple_invalid_keys_raise_validation_error(self):
        """Test that multiple invalid keys raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Example(
                invalid_key1='test1',
                invalid_key2='test2',
                input='hello world'
            )
        
        error = exc_info.value
        assert "Extra inputs are not permitted" in str(error)
        assert "invalid_key1" in str(error)
        assert "invalid_key2" in str(error)

    def test_valid_keys_work_correctly(self):
        """Test that all valid keys work without errors."""
        example = Example(
            input="hello world",
            actual_output="response",
            expected_output="expected response",
            context=["context1", "context2"],
            retrieval_context=["retrieval1"],
            additional_metadata={"key": "value"},
            tools_called=["tool1"],
            name="test example",
            example_index=1,
            trace_id="trace123",
            trace_span_id="span123",
            dataset_id="dataset123"
        )
        
        assert example.input == "hello world"
        assert example.actual_output == "response"
        assert example.expected_output == "expected response"
        assert example.context == ["context1", "context2"]
        assert example.retrieval_context == ["retrieval1"]
        assert example.additional_metadata == {"key": "value"}
        assert example.tools_called == ["tool1"]
        assert example.name == "test example"
        assert example.example_index == 1
        assert example.trace_id == "trace123"
        assert example.trace_span_id == "span123"
        assert example.dataset_id == "dataset123"
        # The __init__ method resets example_id to None. This assertion captures the current behavior.
        assert example.example_id is None

    def test_mixed_valid_and_invalid_keys_raise_error(self):
        """Test that having both valid and invalid keys still raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            Example(
                input="hello world",  # valid
                actual_output="response",  # valid
                invalid_key="test",  # invalid - should cause error
                name="test example"  # valid
            )
        
        error = exc_info.value
        assert "Extra inputs are not permitted" in str(error)
        assert "invalid_key" in str(error)

    def test_typo_in_field_name_raises_error(self):
        """Test that common typos in field names are caught."""
        # Test typo in 'input'
        with pytest.raises(ValidationError):
            Example(inptu='test')  # typo: inptu instead of input
        
        # Test typo in 'actual_output'
        with pytest.raises(ValidationError):
            Example(actual_outpu='test')  # typo: actual_outpu instead of actual_output
        
        # Test typo in 'expected_output'
        with pytest.raises(ValidationError):
            Example(expected_outpu='test')  # typo: expected_outpu instead of expected_output

    def test_case_sensitive_field_names(self):
        """Test that field names are case sensitive."""
        with pytest.raises(ValidationError):
            Example(Input='test')  # wrong case: Input instead of input
        
        with pytest.raises(ValidationError):
            Example(ACTUAL_OUTPUT='test')  # wrong case: ACTUAL_OUTPUT instead of actual_output

    def test_empty_example_creation(self):
        """Test that an empty Example can be created without any fields."""
        example = Example()
        assert example.input is None
        assert example.actual_output is None
        assert example.name is None
        assert example.created_at is not None  # should be auto-set