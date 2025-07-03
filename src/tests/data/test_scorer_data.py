import pytest

from judgeval.data.scorer_data import ScorerData, create_scorer_data
from judgeval.scorers.example_scorer import ExampleScorer


class MockBaseScorer(ExampleScorer):
    def score_example(self, example, *args, **kwargs):
        pass

    async def a_score_example(self, example, *args, **kwargs):
        pass

    def _success_check(self) -> bool:
        return self.score >= self.threshold if self.score is not None else False


@pytest.fixture
def successful_scorer():
    """
    Fixture for a scorer that executes successfully and stores the results of the evaluation
    """
    return MockBaseScorer(
        score_type="test_scorer",
        threshold=0.7,
        score=0.8,
        reason="Test passed successfully",
        evaluation_model="gpt-4",
        strict_mode=False,
        evaluation_cost=0.1,
        verbose_logs="Detailed test logs",
        additional_metadata={"key": "value"},
    )


@pytest.fixture
def failed_scorer():
    """
    Fixture for a scorer that does not pass its threshold expectation
    """
    return MockBaseScorer(
        score_type="test_scorer",
        threshold=0.7,
        score=0.6,
        reason="Test failed",
        evaluation_model="gpt-4",
        strict_mode=False,
        evaluation_cost=0.1,
        verbose_logs="Detailed test logs",
    )


@pytest.fixture
def error_scorer():
    """
    Fixture for a scorer that encounters an error during execution
    """
    return MockBaseScorer(
        score_type="test_scorer",
        threshold=0.7,
        error="Test execution failed",
        evaluation_model="gpt-4",
        evaluation_cost=0.1,
        verbose_logs="Error logs",
    )


def test_scorer_data_successful_case(successful_scorer):
    """Test ScorerData creation for a successful evaluation"""
    scorer_data = create_scorer_data(successful_scorer)

    assert scorer_data.name == "test_scorer"
    assert scorer_data.threshold == 0.7
    assert scorer_data.score == 0.8
    assert scorer_data.success is True
    assert scorer_data.reason == "Test passed successfully"
    assert scorer_data.strict_mode is False
    assert scorer_data.evaluation_model == "gpt-4"
    assert scorer_data.error is None
    assert scorer_data.evaluation_cost == 0.1
    assert scorer_data.verbose_logs == "Detailed test logs"
    assert scorer_data.additional_metadata == {"key": "value"}


def test_scorer_data_failed_case(failed_scorer):
    """Test ScorerData creation for a failed evaluation"""
    scorer_data = create_scorer_data(failed_scorer)

    assert scorer_data.name == "test_scorer"
    assert scorer_data.threshold == 0.7
    assert scorer_data.score == 0.6
    assert scorer_data.success is False
    assert scorer_data.reason == "Test failed"
    assert scorer_data.error is None


def test_scorer_data_error_case(error_scorer):
    """Test ScorerData creation when an error occurs"""
    scorer_data = create_scorer_data(error_scorer)

    assert scorer_data.name == "test_scorer"
    assert scorer_data.threshold == 0.7
    assert scorer_data.score is None
    assert scorer_data.success is False
    assert scorer_data.reason is None
    assert scorer_data.error == "Test execution failed"


def test_scorer_data_to_dict(successful_scorer):
    """Test the to_dict method of ScorerData"""
    scorer_data = create_scorer_data(successful_scorer)
    data_dict = scorer_data.to_dict()

    assert isinstance(data_dict, dict)
    assert data_dict["name"] == "test_scorer"
    assert data_dict["threshold"] == 0.7
    assert data_dict["score"] == 0.8
    assert data_dict["success"] is True
    assert data_dict["reason"] == "Test passed successfully"
    assert data_dict["strict_mode"] is False
    assert data_dict["evaluation_model"] == "gpt-4"
    assert data_dict["error"] is None
    assert data_dict["evaluation_cost"] == 0.1
    assert data_dict["verbose_logs"] == "Detailed test logs"
    assert data_dict["additional_metadata"] == {"key": "value"}


def test_scorer_data_direct_creation():
    """Test direct creation of ScorerData object"""
    scorer_data = ScorerData(
        name="direct_test",
        threshold=0.5,
        success=True,
        score=0.75,
        reason="Direct creation test",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        evaluation_cost=0.2,
        verbose_logs="Test logs",
        additional_metadata={"test": "data"},
    )

    assert scorer_data.name == "direct_test"
    assert scorer_data.threshold == 0.5
    assert scorer_data.success is True
    assert scorer_data.score == 0.75


def test_scorer_data_minimal_creation():
    """Test creation of ScorerData with minimal required fields"""
    scorer_data = ScorerData(name="minimal_test", threshold=0.5, success=True)

    assert scorer_data.name == "minimal_test"
    assert scorer_data.threshold == 0.5
    assert scorer_data.success is True
    assert scorer_data.score is None
    assert scorer_data.reason is None
    assert scorer_data.strict_mode is None
    assert scorer_data.evaluation_model is None
    assert scorer_data.error is None
    assert scorer_data.evaluation_cost is None
    assert scorer_data.verbose_logs is None
    assert scorer_data.additional_metadata is None


def test_scorer_data_to_dict_minimal():
    """Test to_dict method with minimal required fields"""
    scorer_data = ScorerData(name="minimal_test", threshold=0.5, success=True)
    data_dict = scorer_data.to_dict()

    assert isinstance(data_dict, dict)
    assert data_dict["name"] == "minimal_test"
    assert data_dict["threshold"] == 0.5
    assert data_dict["success"] is True
    assert data_dict["score"] is None
    assert data_dict["reason"] is None
    assert data_dict["strict_mode"] is None
    assert data_dict["evaluation_model"] is None
    assert data_dict["error"] is None
    assert data_dict["evaluation_cost"] is None
    assert data_dict["verbose_logs"] is None
    assert data_dict["additional_metadata"] is None


def test_scorer_data_to_dict_with_list_model():
    """Test to_dict method when evaluation_model is a list"""
    scorer_data = ScorerData(
        name="list_model_test",
        threshold=0.5,
        success=True,
        evaluation_model=["gpt-4", "gpt-3.5-turbo"],
    )
    data_dict = scorer_data.to_dict()

    assert isinstance(data_dict["evaluation_model"], list)
    assert data_dict["evaluation_model"] == ["gpt-4", "gpt-3.5-turbo"]


def test_scorer_data_to_dict_with_error():
    """Test to_dict method with error information"""
    scorer_data = ScorerData(
        name="error_test", threshold=0.5, success=False, error="Test error message"
    )
    data_dict = scorer_data.to_dict()

    assert data_dict["error"] == "Test error message"
    assert data_dict["success"] is False
    assert data_dict["score"] is None


def test_scorer_data_to_dict_all_parameters():
    """Test to_dict method with all possible parameters set"""
    test_metadata = {
        "model_tokens": 150,
        "completion_tokens": 50,
        "custom_field": "custom_value",
    }

    scorer_data = ScorerData(
        name="full_test",
        threshold=0.75,
        success=True,
        score=0.85,
        reason="Comprehensive test case",
        strict_mode=True,
        evaluation_model=["gpt-4", "gpt-3.5-turbo"],
        error=None,
        evaluation_cost=0.123,
        verbose_logs="Detailed execution logs\nwith multiple lines",
        additional_metadata=test_metadata,
    )
    data_dict = scorer_data.to_dict()

    # Verify all fields are present and have correct values
    assert isinstance(data_dict, dict)
    assert data_dict["name"] == "full_test"
    assert data_dict["threshold"] == 0.75
    assert data_dict["success"] is True
    assert data_dict["score"] == 0.85
    assert data_dict["reason"] == "Comprehensive test case"
    assert data_dict["strict_mode"] is True
    assert data_dict["evaluation_model"] == ["gpt-4", "gpt-3.5-turbo"]
    assert data_dict["error"] is None
    assert data_dict["evaluation_cost"] == 0.123
    assert data_dict["verbose_logs"] == "Detailed execution logs\nwith multiple lines"
    assert data_dict["additional_metadata"] == test_metadata

    # Verify the metadata dictionary contains all expected fields
    assert data_dict["additional_metadata"]["model_tokens"] == 150
    assert data_dict["additional_metadata"]["completion_tokens"] == 50
    assert data_dict["additional_metadata"]["custom_field"] == "custom_value"
