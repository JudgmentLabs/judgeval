import pytest
from judgeval.data import Example, ProcessExample, ScorerData
from judgeval.data.api_example import create_process_example

# Test data fixtures
@pytest.fixture
def basic_example():
    return Example(
        name="test_case",
        input="test input",
        actual_output="actual output",
        expected_output="expected output"
    )

@pytest.fixture
def basic_scorer_data():
    return ScorerData(
        name="test_scorer",
        threshold=1.0,
        success=True,
        score=1.0,
        metadata={"key": "value"}
    )

class TestProcessExample:
    def test_create_basic_process_example(self):
        """Test creating a basic ProcessExample with required fields"""
        process_ex = ProcessExample(
            name="test",
            input="test input",
            actual_output="test output"
        )
        assert process_ex.name == "test"
        assert process_ex.input == "test input"
        assert process_ex.actual_output == "test output"

    def test_validation_error_missing_input(self):
        """Test validation error when input is missing"""
        with pytest.raises(ValueError) as exc_info:
            ProcessExample(
                name="test",
                actual_output="test output"
            )
        assert "'input' and 'actual_output' must be provided" in str(exc_info.value)

    def test_validation_error_missing_actual_output(self):
        """Test validation error when actual_output is missing"""
        with pytest.raises(ValueError) as exc_info:
            ProcessExample(
                name="test",
                input="test input"
            )
        assert "'input' and 'actual_output' must be provided" in str(exc_info.value)

    def test_update_scorer_data_initial(self, basic_scorer_data):
        """Test updating scorer data for the first time"""
        process_ex = ProcessExample(
            name="test",
            input="test input",
            actual_output="test output"
        )
        process_ex.update_scorer_data(basic_scorer_data)
        
        assert process_ex.success == True
        assert len(process_ex.scorers_data) == 1
        assert process_ex.scorers_data[0] == basic_scorer_data

    def test_update_scorer_data_multiple(self, basic_scorer_data):
        """Test updating scorer data multiple times"""
        process_ex = ProcessExample(
            name="test",
            input="test input",
            actual_output="test output"
        )
        
        # Add first scorer
        process_ex.update_scorer_data(basic_scorer_data)
        
        # Add second scorer with failure
        failed_scorer = ScorerData(
            name="failed_scorer",
            threshold=1.0,
            success=False,
            score=0.0,
            metadata={}
        )
        process_ex.update_scorer_data(failed_scorer)
        
        assert process_ex.success == False
        assert len(process_ex.scorers_data) == 2
        assert process_ex.scorers_data[1] == failed_scorer

    def test_update_run_duration(self):
        """Test updating run duration"""
        process_ex = ProcessExample(
            name="test",
            input="test input",
            actual_output="test output"
        )
        process_ex.update_run_duration(1.5)
        assert process_ex.run_duration == 1.5

class TestCreateProcessExample:
    def test_create_process_example_basic(self, basic_example):
        """Test creating ProcessExample from basic Example"""
        process_ex = create_process_example(basic_example)
        
        assert process_ex.name == "test_case"
        assert process_ex.input == "test input"
        assert process_ex.actual_output == "actual output"
        assert process_ex.expected_output == "expected output"
        assert process_ex.success == True
        assert process_ex.scorers_data == []
        assert process_ex.run_duration is None
        assert process_ex.evaluation_cost is None

    def test_create_process_example_no_name(self):
        """Test creating ProcessExample from Example without name"""
        example = Example(
            input="test input",
            actual_output="actual output"
        )
        process_ex = create_process_example(example)
        
        assert process_ex.name == "Test Case Placeholder"
        assert process_ex.input == "test input"
        assert process_ex.actual_output == "actual output"

    def test_create_process_example_with_all_fields(self):
        """Test creating ProcessExample with all possible fields"""
        example = Example(
            name="full_test",
            input="test input",
            actual_output="actual output",
            expected_output="expected output",
            context=["context1", "context2"],
            retrieval_context=["retrieval1", "retrieval2"],
            tools_called=["tool1", "tool2"],
            expected_tools=["expected_tool1"],
            additional_metadata={"key": "value"},
            trace_id="trace123"
        )
        
        process_ex = create_process_example(example)
        
        assert process_ex.name == "full_test"
        assert process_ex.context == ["context1", "context2"]
        assert process_ex.retrieval_context == ["retrieval1", "retrieval2"]
        assert process_ex.tools_called == ["tool1", "tool2"]
        assert process_ex.expected_tools == ["expected_tool1"]
        assert process_ex.additional_metadata == {"key": "value"}
        assert process_ex.trace_id == "trace123"
