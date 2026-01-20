from judgeval.v1.scorers.built_in.faithfulness import FaithfulnessScorer
from judgeval.constants import APIScorerType


def test_faithfulness_initialization():
    scorer = FaithfulnessScorer()
    assert scorer._threshold == 0.5
    assert scorer._name == APIScorerType.FAITHFULNESS.value
    assert scorer._model is None


def test_faithfulness_with_custom_threshold():
    scorer = FaithfulnessScorer(threshold=0.9)
    assert scorer._threshold == 0.9


def test_faithfulness_with_name():
    scorer = FaithfulnessScorer(name="Custom Faithfulness Scorer")
    assert scorer._name == "Custom Faithfulness Scorer"


def test_faithfulness_with_model():
    scorer = FaithfulnessScorer(model="gpt-4")
    assert scorer._model == "gpt-4"


def test_faithfulness_required_params():
    scorer = FaithfulnessScorer()
    assert scorer._required_params == ["context", "actual_output"]


def test_faithfulness_score_type():
    scorer = FaithfulnessScorer()
    assert scorer._score_type == APIScorerType.FAITHFULNESS.value


def test_faithfulness_create_method():
    scorer = FaithfulnessScorer.create(threshold=0.85)
    assert isinstance(scorer, FaithfulnessScorer)
    assert scorer._threshold == 0.85


def test_faithfulness_get_scorer_config():
    scorer = FaithfulnessScorer(
        threshold=1.0, name="Strict Faithfulness", model="gpt-4o"
    )

    config = scorer.get_scorer_config()

    assert config["score_type"] == APIScorerType.FAITHFULNESS.value
    assert config.get("threshold") == 1.0
    assert config.get("name") == "Strict Faithfulness"
    assert config.get("kwargs", {}).get("model") == "gpt-4o"
