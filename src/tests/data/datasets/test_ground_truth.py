import pytest
from judgeval.data import GroundTruthExample


def test_ground_truth_example_minimal():
    """Test creation with only required field (input)"""
    example = GroundTruthExample(input="test input")
    assert example.input == "test input"
    assert example.actual_output is None
    assert example.expected_output is None


def test_ground_truth_example_full():
    """Test creation with all fields populated"""
    example = GroundTruthExample(
        input="test input",
        actual_output="actual result",
        expected_output="expected result",
        context=["context1", "context2"],
        retrieval_context=["retrieved1", "retrieved2"],
        additional_metadata={"key": "value"},
        comments="test comment",
        tools_called=["tool1", "tool2"],
        expected_tools=["expected_tool1"],
        source_file="test.txt"
    )
    
    assert example.input == "test input"
    assert example.actual_output == "actual result"
    assert example.expected_output == "expected result"
    assert example.context == ["context1", "context2"]
    assert example.retrieval_context == ["retrieved1", "retrieved2"]
    assert example.additional_metadata == {"key": "value"}
    assert example.comments == "test comment"
    assert example.tools_called == ["tool1", "tool2"]
    assert example.expected_tools == ["expected_tool1"]
    assert example.source_file == "test.txt"


def test_ground_truth_example_to_dict():
    """Test the to_dict method returns correct dictionary"""
    example = GroundTruthExample(
        input="test input",
        actual_output="actual result",
        comments="test comment"
    )
    
    expected_dict = {
        "input": "test input",
        "actual_output": "actual result",
        "expected_output": None,
        "context": None,
        "retrieval_context": None,
        "additional_metadata": None,
        "comments": "test comment",
        "tools_called": None,
        "expected_tools": None,
        "source_file": None,
        "trace_id": None
    }
    
    assert example.to_dict() == expected_dict


def test_ground_truth_example_str_representation():
    """Test the string representation of the class"""
    example = GroundTruthExample(
        input="test input",
        actual_output="actual result"
    )
    
    expected_str = (
        "GroundTruthExample("
        "input=test input, "
        "actual_output=actual result, "
        "expected_output=None, "
        "context=None, "
        "retrieval_context=None, "
        "additional_metadata=None, "
        "comments=None, "
        "tools_called=None, "
        "expected_tools=None, "
        "source_file=None, "
        "trace_id=None)"
    )
    
    assert str(example) == expected_str


def test_ground_truth_example_missing_input():
    """Test that creating instance without required 'input' field raises error"""
    with pytest.raises(ValueError):
        GroundTruthExample()


def test_ground_truth_example_invalid_types():
    """Test that invalid types raise validation errors"""
    with pytest.raises(ValueError):
        GroundTruthExample(input="test", context="not a list")
    
    with pytest.raises(ValueError):
        GroundTruthExample(input="test", tools_called="not a list")
    
    with pytest.raises(ValueError):
        GroundTruthExample(input="test", additional_metadata="not a dict")


def test_ground_truth_example_empty_lists():
    """Test that empty lists are valid for list fields"""
    example = GroundTruthExample(
        input="test",
        context=[],
        retrieval_context=[],
        tools_called=[],
        expected_tools=[]
    )
    assert example.context == []
    assert example.retrieval_context == []
    assert example.tools_called == []
    assert example.expected_tools == []


def test_ground_truth_example_empty_dict():
    """Test that empty dict is valid for additional_metadata"""
    example = GroundTruthExample(
        input="test",
        additional_metadata={}
    )
    assert example.additional_metadata == {}
    