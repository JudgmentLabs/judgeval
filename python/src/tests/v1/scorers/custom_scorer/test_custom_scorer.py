import pytest
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer


def test_custom_scorer_initialization():
    scorer = CustomScorer(name="TestScorer")
    assert scorer._name == "TestScorer"
    assert scorer._class_name == "TestScorer"
    assert scorer._server_hosted is True


def test_custom_scorer_with_class_name():
    scorer = CustomScorer(name="DisplayName", class_name="ActualClassName")
    assert scorer._name == "DisplayName"
    assert scorer._class_name == "ActualClassName"


def test_custom_scorer_not_server_hosted():
    scorer = CustomScorer(name="LocalScorer", server_hosted=False)
    assert scorer._server_hosted is False


def test_custom_scorer_get_name():
    scorer = CustomScorer(name="MyScorer")
    assert scorer.get_name() == "MyScorer"


def test_custom_scorer_get_class_name():
    scorer = CustomScorer(name="DisplayName", class_name="ClassName")
    assert scorer.get_class_name() == "ClassName"


def test_custom_scorer_is_server_hosted():
    scorer = CustomScorer(name="TestScorer", server_hosted=True)
    assert scorer.is_server_hosted() is True

    local_scorer = CustomScorer(name="LocalScorer", server_hosted=False)
    assert local_scorer.is_server_hosted() is False


def test_custom_scorer_get_scorer_config_raises():
    scorer = CustomScorer(name="TestScorer")
    with pytest.raises(NotImplementedError):
        scorer.get_scorer_config()
