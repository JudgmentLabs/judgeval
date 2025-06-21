import pytest
from pydantic import ValidationError

from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer


@pytest.fixture
def valid_scorer_params():
    return {"threshold": 0.8, "score_type": APIScorer.FAITHFULNESS}


def test_judgment_scorer_creation_with_enum():
    """Test creating JudgmentScorer with APIScorer enum value"""
    scorer = APIJudgmentScorer(threshold=0.8, score_type=APIScorer.FAITHFULNESS)
    assert scorer.threshold == 0.8
    assert scorer.score_type == "faithfulness"


def test_judgment_scorer_creation_with_string():
    """Test creating JudgmentScorer with string value"""
    scorer = APIJudgmentScorer(threshold=0.8, score_type="faithfulness")
    assert scorer.threshold == 0.8
    assert scorer.score_type == "faithfulness"


def test_judgment_scorer_creation_with_uppercase_string():
    """Test creating JudgmentScorer with uppercase string value"""
    scorer = APIJudgmentScorer(threshold=0.8, score_type="FAITHFULNESS")
    assert scorer.threshold == 0.8
    assert scorer.score_type == "faithfulness"


def test_judgment_scorer_str_representation():
    """Test the string representation of JudgmentScorer"""
    scorer = APIJudgmentScorer(threshold=0.8, score_type=APIScorer.FAITHFULNESS)
    expected_str = "JudgmentScorer(score_type=faithfulness, threshold=0.8)"
    assert str(scorer) == expected_str


@pytest.mark.parametrize(
    "invalid_score_type",
    [
        123,  # integer
        None,  # None
        True,  # boolean
        ["faithfulness"],  # list
        {"type": "faithfulness"},  # dict
    ],
)
def test_judgment_scorer_invalid_score_type(invalid_score_type):
    """Test creating JudgmentScorer with invalid score_type values"""
    with pytest.raises(ValidationError) as exc_info:
        APIJudgmentScorer(threshold=0.8, score_type=invalid_score_type)

    assert "Input should be" in str(exc_info.value)


def test_judgment_scorer_invalid_string_value():
    """Test creating JudgmentScorer with invalid string value"""
    with pytest.raises(ValidationError):
        APIJudgmentScorer(threshold=0.8, score_type="INVALID_METRIC")


def test_judgment_scorer_threshold_validation():
    """Test threshold validation"""
    # Test float values
    scorer = APIJudgmentScorer(threshold=0.5, score_type=APIScorer.FAITHFULNESS)
    assert scorer.threshold == 0.5

    # Test integer values (should be converted to float)
    scorer = APIJudgmentScorer(threshold=1, score_type=APIScorer.FAITHFULNESS)
    assert scorer.threshold == 1.0
