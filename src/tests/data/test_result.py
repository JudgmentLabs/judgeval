import pytest
from judgeval.data import ScorerData, Example, ScoringResult
from judgeval.data.result import generate_scoring_result


@pytest.fixture
def sample_scorer_data():
    return ScorerData(
        name="test_scorer",
        threshold=1.0,
        success=True,
        score=0.8,
        additional_metadata={"key": "value"},
    )


@pytest.fixture
def sample_example():
    return Example(
        name="test_example",
        input="test input",
        actual_output="actual output",
        expected_output="expected output",
        context=["context1", "context2"],
        retrieval_context=["retrieval1"],
    )


class TestScoringResult:
    def test_basic_initialization(self):
        """Test basic initialization with minimal required fields"""
        result = ScoringResult(success=True, scorers_data=[], data_object=Example())
        assert result.success is True
        assert result.scorers_data == []
        assert result.data_object.input is None
        assert result.data_object.actual_output is None

    def test_full_initialization(self, sample_scorer_data, sample_example):
        """Test initialization with all fields"""
        result = ScoringResult(
            success=True,
            scorers_data=[sample_scorer_data],
            data_object=sample_example,
            trace_id="trace123",
        )

        assert result.success is True
        assert len(result.scorers_data) == 1
        assert result.data_object.input == "test input"
        assert result.data_object.actual_output == "actual output"
        assert result.data_object.expected_output == "expected output"
        assert result.data_object.context == ["context1", "context2"]
        assert result.data_object.retrieval_context == ["retrieval1"]
        assert result.trace_id == "trace123"

    def test_to_dict_conversion(self, sample_scorer_data, sample_example):
        """Test conversion to dictionary"""
        result = ScoringResult(
            success=True, scorers_data=[sample_scorer_data], data_object=sample_example
        )

        dict_result = result.to_dict()
        assert isinstance(dict_result, dict)
        assert dict_result["success"] is True
        assert len(dict_result["scorers_data"]) == 1
        assert dict_result["data_object"]["input"] == "test input"
        assert dict_result["data_object"]["actual_output"] == "actual output"

    def test_to_dict_with_none_scorers(self):
        """Test conversion to dictionary when scorers_data is None"""
        result = ScoringResult(success=False, scorers_data=None)
        dict_result = result.to_dict()
        assert dict_result["scorers_data"] is None

    def test_string_representation(self, sample_scorer_data):
        """Test string representation of ScoringResult"""
        result = ScoringResult(success=True, scorers_data=[sample_scorer_data])
        str_result = str(result)
        assert "ScoringResult" in str_result
        assert "success=True" in str_result


class TestGenerateScoringResult:
    def test_generate_from_example(self, sample_example):
        """Test generating ScoringResult from Example"""
        result = generate_scoring_result(sample_example, [], 0.0, True)

        assert isinstance(result, ScoringResult)
        assert result.data_object.input == sample_example.input
        assert result.data_object.actual_output == sample_example.actual_output
        assert result.data_object.expected_output == sample_example.expected_output
        assert result.data_object.context == sample_example.context
        assert result.data_object.retrieval_context == sample_example.retrieval_context
        assert result.trace_id == sample_example.trace_id

    def test_generate_with_minimal_example(self):
        """Test generating ScoringResult from minimal Example"""
        minimal_example = Example(
            name="minimal",
            input="test",
            actual_output="output",
        )

        result = generate_scoring_result(minimal_example, [], 0.0, True)
        assert isinstance(result, ScoringResult)
        assert result.success is True
        assert result.scorers_data == []
        assert result.data_object.input == "test"
        assert result.data_object.actual_output == "output"
        assert result.data_object.expected_output is None
        assert result.data_object.context is None
        assert result.data_object.retrieval_context is None
        assert result.trace_id is None
