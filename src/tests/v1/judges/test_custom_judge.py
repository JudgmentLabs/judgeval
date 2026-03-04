import pytest
from judgeval.v1.judges.custom_judge import CustomJudge

TEST_PROJECT_ID = "test-project-id"


def test_custom_judge_initialization():
    judge = CustomJudge(name="TestJudge", project_id=TEST_PROJECT_ID)
    assert judge._name == "TestJudge"
    assert judge._project_id == TEST_PROJECT_ID


def test_custom_judge_get_name():
    judge = CustomJudge(name="MyJudge", project_id=TEST_PROJECT_ID)
    assert judge.get_name() == "MyJudge"


def test_custom_judge_get_scorer_config_raises():
    judge = CustomJudge(name="TestJudge", project_id=TEST_PROJECT_ID)
    with pytest.raises(NotImplementedError):
        judge.get_scorer_config()
