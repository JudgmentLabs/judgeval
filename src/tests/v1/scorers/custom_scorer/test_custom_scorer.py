import pytest
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer

TEST_PROJECT_ID = "test-project-id"


def test_custom_scorer_initialization():
    scorer = CustomScorer(name="TestScorer", project_id=TEST_PROJECT_ID)
    assert scorer._name == "TestScorer"
    assert scorer._class_name == "TestScorer"
    assert scorer._server_hosted is True
    assert scorer._project_id == TEST_PROJECT_ID


def test_custom_scorer_with_class_name():
    scorer = CustomScorer(
        name="DisplayName", class_name="ActualClassName", project_id=TEST_PROJECT_ID
    )
    assert scorer._name == "DisplayName"
    assert scorer._class_name == "ActualClassName"


def test_custom_scorer_not_server_hosted():
    scorer = CustomScorer(
        name="LocalScorer", server_hosted=False, project_id=TEST_PROJECT_ID
    )
    assert scorer._server_hosted is False


def test_custom_scorer_get_name():
    scorer = CustomScorer(name="MyScorer", project_id=TEST_PROJECT_ID)
    assert scorer.get_name() == "MyScorer"


def test_custom_scorer_get_class_name():
    scorer = CustomScorer(
        name="DisplayName", class_name="ClassName", project_id=TEST_PROJECT_ID
    )
    assert scorer.get_class_name() == "ClassName"


def test_custom_scorer_is_server_hosted():
    scorer = CustomScorer(
        name="TestScorer", server_hosted=True, project_id=TEST_PROJECT_ID
    )
    assert scorer.is_server_hosted() is True

    local_scorer = CustomScorer(
        name="LocalScorer", server_hosted=False, project_id=TEST_PROJECT_ID
    )
    assert local_scorer.is_server_hosted() is False


def test_custom_scorer_get_scorer_config_raises():
    scorer = CustomScorer(name="TestScorer", project_id=TEST_PROJECT_ID)
    with pytest.raises(NotImplementedError):
        scorer.get_scorer_config()
