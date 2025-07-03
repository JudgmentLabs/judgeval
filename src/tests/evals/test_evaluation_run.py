import pytest
from pydantic import ValidationError

from judgeval.evaluation_run import EvaluationRun
from judgeval.data import Example, CustomExample
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.judges import JudgevalJudge
from judgeval.constants import APIScorerType


class MockScorer(BaseScorer):
    def __init__(
        self, score_type: str = APIScorerType.FAITHFULNESS, threshold: float = 0.5
    ):
        super().__init__(score_type=score_type, threshold=threshold)


class MockAPIScorer(APIScorerConfig):
    def __init__(
        self, score_type: str = APIScorerType.FAITHFULNESS, threshold: float = 0.5
    ):
        super().__init__(score_type=score_type, threshold=threshold)


class MockJudge(JudgevalJudge):
    def __init__(self):
        super().__init__()
        self.name = "mock-judge"
        self.model = "mock-model"

    def load_model(self) -> None:
        """Mock implementation of load_model."""
        pass

    def generate(self, prompt: str) -> str:
        """Mock implementation of generate."""
        return '{"score": 1, "reason": "mock response"}'

    async def a_generate(self, prompt: str) -> str:
        """Mock implementation of a_generate."""
        return '{"score": 1, "reason": "mock response"}'

    def get_model_name(self) -> str:
        """Mock implementation of get_model_name."""
        return "mock-model"


def test_validate_examples():
    # Test valid examples
    examples = [
        Example(input="test1", actual_output="test1"),
        Example(input="test2", actual_output="test2"),
    ]
    run = EvaluationRun(examples=examples, scorers=[MockScorer()])
    assert run.examples == examples

    # Test empty examples
    with pytest.raises(ValueError, match="Examples cannot be empty"):
        EvaluationRun(examples=[], scorers=[MockScorer()])

    # Test mixed example types
    with pytest.raises(ValidationError):
        EvaluationRun(
            examples=[
                Example(input="test1", actual_output="test1"),
                CustomExample(
                    input={"question": "test2"}, actual_output={"answer": "test2"}
                ),
            ],
            scorers=[MockScorer()],
        )


def test_validate_scorers():
    # Test valid scorers
    scorers = [MockScorer(), MockAPIScorer()]
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")], scorers=scorers
    )
    assert run.scorers == scorers

    # Test empty scorers
    with pytest.raises(ValueError, match="Scorers cannot be empty"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")], scorers=[]
        )

    # Test invalid scorer type
    class InvalidScorer:
        pass

    with pytest.raises(
        ValueError, match="All scorers must be of type BaseScorer or APIScorerConfig."
    ):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[InvalidScorer()],
        )


def test_validate_model():
    # Test valid string model
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        model="gpt-4.1",
    )
    assert run.model == "gpt-4.1"

    # Test invalid model name
    with pytest.raises(ValueError, match="Model name invalid-model not recognized"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            model="invalid-model",
        )

    # Test invalid model type
    with pytest.raises(ValidationError):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            model=123,
        )

    # Test JudgevalJudge with APIJudgmentScorer
    with pytest.raises(
        ValueError,
    ):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockAPIScorer()],
            model=MockJudge(),
        )
