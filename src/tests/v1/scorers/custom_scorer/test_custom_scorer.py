import pytest
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer

TEST_PROJECT_ID = "test-project-id"


def test_custom_scorer_initialization():
    scorer = CustomScorer(name="TestScorer", project_id=TEST_PROJECT_ID)
    assert scorer._name == "TestScorer"
    assert scorer._project_id == TEST_PROJECT_ID


def test_custom_scorer_get_name():
    scorer = CustomScorer(name="MyScorer", project_id=TEST_PROJECT_ID)
    assert scorer.get_name() == "MyScorer"


def test_custom_scorer_get_scorer_config_raises():
    scorer = CustomScorer(name="TestScorer", project_id=TEST_PROJECT_ID)
    with pytest.raises(NotImplementedError):
        scorer.get_scorer_config()
