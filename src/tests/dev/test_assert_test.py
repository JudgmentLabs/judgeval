import pytest
from judgeval.data import ScoringResult, ScorerData, Example
from judgeval.run_evaluation import assert_test


@pytest.fixture
def sample_example():
    return Example(
        name="test_example",
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context=["context1", "context2"],
        retrieval_context=["retrieval1"],
    )


def test_assert_test_all_passing(sample_example):
    """Test when all results are successful"""
    scorer_data = ScorerData(
        name="test_scorer",
        success=True,
        score=0.9,
        threshold=0.8,
        reason="Test passed",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        additional_metadata={},
    )

    result = ScoringResult(
        data_object=sample_example, success=True, scorers_data=[scorer_data]
    )

    # Should not raise any exception
    assert_test([result])


def test_assert_test_failed_scorer(sample_example):
    """Test when a scorer fails"""
    failed_scorer = ScorerData(
        name="failed_scorer",
        success=False,
        score=0.5,
        threshold=0.8,
        reason="Score below threshold",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        additional_metadata={},
    )

    result = ScoringResult(
        data_object=sample_example, success=False, scorers_data=[failed_scorer]
    )

    with pytest.raises(AssertionError) as exc_info:
        assert_test([result])

    # Verify error message contains relevant information
    error_msg = str(exc_info.value)
    assert "failed_scorer" in error_msg
    assert "Score below threshold" in error_msg


def test_assert_test_multiple_failed_scorers(sample_example):
    """Test when multiple scorers fail"""
    failed_scorer1 = ScorerData(
        name="scorer1",
        success=False,
        score=0.5,
        threshold=0.8,
        reason="First failure",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        additional_metadata={},
    )

    failed_scorer2 = ScorerData(
        name="scorer2",
        success=False,
        score=0.6,
        threshold=0.8,
        reason="Second failure",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        additional_metadata={},
    )

    result = ScoringResult(
        data_object=sample_example,
        success=False,
        scorers_data=[failed_scorer1, failed_scorer2],
    )

    with pytest.raises(AssertionError) as exc_info:
        assert_test([result])

    error_msg = str(exc_info.value)
    assert "scorer1" in error_msg
    assert "scorer2" in error_msg
    assert "First failure" in error_msg
    assert "Second failure" in error_msg


def test_assert_test_empty_results():
    """Test with empty results list"""
    # Should not raise any exception
    assert_test([])
