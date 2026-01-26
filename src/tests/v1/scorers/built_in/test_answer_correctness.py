from judgeval.v1.scorers.built_in.answer_correctness import AnswerCorrectnessScorer
from judgeval.constants import APIScorerType


def test_answer_correctness_initialization():
    scorer = AnswerCorrectnessScorer()
    assert scorer._threshold == 0.5
    assert scorer._name == APIScorerType.ANSWER_CORRECTNESS.value
    assert scorer._model is None


def test_answer_correctness_with_custom_threshold():
    scorer = AnswerCorrectnessScorer(threshold=0.8)
    assert scorer._threshold == 0.8


def test_answer_correctness_with_name():
    scorer = AnswerCorrectnessScorer(name="Custom AC Scorer")
    assert scorer._name == "Custom AC Scorer"


def test_answer_correctness_with_model():
    scorer = AnswerCorrectnessScorer(model="gpt-4")
    assert scorer._model == "gpt-4"


def test_answer_correctness_required_params():
    scorer = AnswerCorrectnessScorer()
    assert scorer._required_params == ["input", "actual_output", "expected_output"]


def test_answer_correctness_score_type():
    scorer = AnswerCorrectnessScorer()
    assert scorer._score_type == APIScorerType.ANSWER_CORRECTNESS.value


def test_answer_correctness_create_method():
    scorer = AnswerCorrectnessScorer.create(threshold=0.7)
    assert isinstance(scorer, AnswerCorrectnessScorer)
    assert scorer._threshold == 0.7


def test_answer_correctness_get_scorer_config():
    scorer = AnswerCorrectnessScorer(threshold=0.6, name="Test Scorer", model="gpt-4")

    config = scorer.get_scorer_config()

    assert config["score_type"] == APIScorerType.ANSWER_CORRECTNESS.value
    assert config.get("threshold") == 0.6
    assert config.get("name") == "Test Scorer"
    assert config.get("kwargs", {}).get("model") == "gpt-4"
