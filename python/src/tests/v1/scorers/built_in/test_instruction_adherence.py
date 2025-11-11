from judgeval.v1.scorers.built_in.instruction_adherence import (
    InstructionAdherenceScorer,
)
from judgeval.constants import APIScorerType


def test_instruction_adherence_initialization():
    scorer = InstructionAdherenceScorer()
    assert scorer._threshold == 0.5
    assert scorer._name == APIScorerType.INSTRUCTION_ADHERENCE.value
    assert scorer._strict_mode is False
    assert scorer._model is None


def test_instruction_adherence_with_custom_threshold():
    scorer = InstructionAdherenceScorer(threshold=0.75)
    assert scorer._threshold == 0.75


def test_instruction_adherence_with_name():
    scorer = InstructionAdherenceScorer(name="Custom IA Scorer")
    assert scorer._name == "Custom IA Scorer"


def test_instruction_adherence_with_strict_mode():
    scorer = InstructionAdherenceScorer(strict_mode=True)
    assert scorer._strict_mode is True


def test_instruction_adherence_with_model():
    scorer = InstructionAdherenceScorer(model="claude-3-sonnet")
    assert scorer._model == "claude-3-sonnet"


def test_instruction_adherence_required_params():
    scorer = InstructionAdherenceScorer()
    assert scorer._required_params == ["input", "actual_output"]


def test_instruction_adherence_score_type():
    scorer = InstructionAdherenceScorer()
    assert scorer._score_type == APIScorerType.INSTRUCTION_ADHERENCE.value


def test_instruction_adherence_create_method():
    scorer = InstructionAdherenceScorer.create(threshold=0.6)
    assert isinstance(scorer, InstructionAdherenceScorer)
    assert scorer._threshold == 0.6


def test_instruction_adherence_get_scorer_config():
    scorer = InstructionAdherenceScorer(
        threshold=0.8, name="Strict Instruction", strict_mode=True, model="gpt-4-turbo"
    )

    config = scorer.get_scorer_config()

    assert config["score_type"] == APIScorerType.INSTRUCTION_ADHERENCE.value
    assert config.get("threshold") == 0.8
    assert config.get("name") == "Strict Instruction"
    assert config.get("strict_mode") is True
    assert config.get("kwargs", {}).get("model") == "gpt-4-turbo"
