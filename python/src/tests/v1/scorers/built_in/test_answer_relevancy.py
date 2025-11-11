from judgeval.v1.scorers.built_in.answer_relevancy import AnswerRelevancyScorer
from judgeval.constants import APIScorerType


def test_answer_relevancy_initialization():
    scorer = AnswerRelevancyScorer()
    assert scorer._threshold == 0.5
    assert scorer._name == APIScorerType.ANSWER_RELEVANCY.value
    assert scorer._strict_mode is False
    assert scorer._model is None


def test_answer_relevancy_with_custom_threshold():
    scorer = AnswerRelevancyScorer(threshold=0.7)
    assert scorer._threshold == 0.7


def test_answer_relevancy_with_name():
    scorer = AnswerRelevancyScorer(name="Custom AR Scorer")
    assert scorer._name == "Custom AR Scorer"


def test_answer_relevancy_with_strict_mode():
    scorer = AnswerRelevancyScorer(strict_mode=True)
    assert scorer._strict_mode is True


def test_answer_relevancy_with_model():
    scorer = AnswerRelevancyScorer(model="gpt-4o-mini")
    assert scorer._model == "gpt-4o-mini"


def test_answer_relevancy_required_params():
    scorer = AnswerRelevancyScorer()
    assert scorer._required_params == ["input", "actual_output"]


def test_answer_relevancy_score_type():
    scorer = AnswerRelevancyScorer()
    assert scorer._score_type == APIScorerType.ANSWER_RELEVANCY.value


def test_answer_relevancy_create_method():
    scorer = AnswerRelevancyScorer.create(threshold=0.8)
    assert isinstance(scorer, AnswerRelevancyScorer)
    assert scorer._threshold == 0.8


def test_answer_relevancy_get_scorer_config():
    scorer = AnswerRelevancyScorer(
        threshold=0.6,
        name="Test Relevancy Scorer",
        strict_mode=False,
        model="claude-3-haiku",
    )

    config = scorer.get_scorer_config()

    assert config["score_type"] == APIScorerType.ANSWER_RELEVANCY.value
    assert config.get("threshold") == 0.6
    assert config.get("name") == "Test Relevancy Scorer"
    assert config.get("strict_mode") is False
    assert config.get("kwargs", {}).get("model") == "claude-3-haiku"
