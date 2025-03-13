import pytest
from judgeval.data import ScorerData, ProcessExample, ScoringResult
from judgeval.data.result import generate_scoring_result

@pytest.fixture
def sample_scorer_data():
    return ScorerData(
        name="test_scorer",
        threshold=1.0,
        success=True,
        score=0.8,
        metadata={"key": "value"}
    )

@pytest.fixture
def sample_process_example(sample_scorer_data):
    return ProcessExample(
        name="test_example",
        input="test input",
        actual_output="actual output",
        expected_output="expected output",
        context=["context1", "context2"],
        retrieval_context=["retrieval1"],
        success=True,
        scorers_data=[sample_scorer_data]
    )

class TestScoringResult:
    def test_basic_initialization(self):
        """Test basic initialization with minimal required fields"""
        result = ScoringResult(success=True, scorers_data=[])
        assert result.success is True
        assert result.scorers_data == []
        assert result.input is None
        assert result.actual_output is None

    def test_full_initialization(self, sample_scorer_data):
        """Test initialization with all fields"""
        result = ScoringResult(
            success=True,
            scorers_data=[sample_scorer_data],
            input="test input",
            actual_output="actual output",
            expected_output="expected output",
            context=["context"],
            retrieval_context=["retrieval"],
            trace_id="trace123"
        )
        
        assert result.success is True
        assert len(result.scorers_data) == 1
        assert result.input == "test input"
        assert result.actual_output == "actual output"
        assert result.expected_output == "expected output"
        assert result.context == ["context"]
        assert result.retrieval_context == ["retrieval"]
        assert result.trace_id == "trace123"

    def test_to_dict_conversion(self, sample_scorer_data):
        """Test conversion to dictionary"""
        result = ScoringResult(
            success=True,
            scorers_data=[sample_scorer_data],
            input="test"
        )
        
        dict_result = result.to_dict()
        assert isinstance(dict_result, dict)
        assert dict_result["success"] is True
        assert len(dict_result["scorers_data"]) == 1
        assert dict_result["input"] == "test"
        assert dict_result["actual_output"] is None

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
    def test_generate_from_process_example(self, sample_process_example):
        """Test generating ScoringResult from ProcessExample"""
        result = generate_scoring_result(sample_process_example)
        
        assert isinstance(result, ScoringResult)
        assert result.success == sample_process_example.success
        assert result.input == sample_process_example.input
        assert result.actual_output == sample_process_example.actual_output
        assert result.expected_output == sample_process_example.expected_output
        assert result.context == sample_process_example.context
        assert result.retrieval_context == sample_process_example.retrieval_context
        assert result.trace_id == sample_process_example.trace_id

    def test_generate_with_minimal_process_example(self):
        """Test generating ScoringResult from minimal ProcessExample"""
        minimal_example = ProcessExample(
            name="minimal",
            input="test",
            actual_output="output",
            success=True,
            scorers_data=[]
        )
        
        result = generate_scoring_result(minimal_example)
        assert isinstance(result, ScoringResult)
        assert result.success is True
        assert result.scorers_data == []
        assert result.input == "test"
        assert result.actual_output == "output"
        assert result.expected_output is None
        assert result.context is None
        assert result.retrieval_context is None
        assert result.trace_id is None
