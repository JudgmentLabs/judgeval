import pytest
from typing import List, Dict, Any
from pydantic import ValidationError

from judgeval.evaluation_run import EvaluationRun
from judgeval.data import Example, CustomExample
from judgeval.scorers import JudgevalScorer, APIJudgmentScorer
from judgeval.judges import JudgevalJudge
from judgeval.rules import Rule

class MockScorer(JudgevalScorer):
    def __init__(self, score_type: str = "faithfulness", threshold: float = 0.5):
        super().__init__(score_type=score_type, threshold=threshold)

class MockAPIScorer(APIJudgmentScorer):
    def __init__(self, score_type: str = "faithfulness", threshold: float = 0.5):
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

def test_validate_log_results():
    # Test valid boolean
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        log_results=True,
        project_name="test-project",
        eval_name="test-eval"
    )
    assert run.log_results is True

    # Test invalid type
    with pytest.raises(ValueError, match="log_results must be a boolean"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            log_results="true"
        )

def test_validate_project_name():
    # Test valid project name when logging
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        log_results=True,
        project_name="test-project",
        eval_name="test-eval"
    )
    assert run.project_name == "test-project"

    # Test missing project name when logging
    with pytest.raises(ValueError, match="Project name is required when log_results is True"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            log_results=True
        )

    # Test no project name when not logging (should pass)
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        log_results=False
    )
    assert run.project_name is None

def test_validate_eval_name():
    # Test valid eval name when logging
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        log_results=True,
        eval_name="test-eval",
        project_name="test-project"
    )
    assert run.eval_name == "test-eval"

    # Test missing eval name when logging
    with pytest.raises(ValueError, match="Eval name is required when log_results is True"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            log_results=True
        )

    # Test no eval name when not logging (should pass)
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        log_results=False
    )
    assert run.eval_name is None

def test_validate_examples():
    # Test valid examples
    examples = [
        Example(input="test1", actual_output="test1"),
        Example(input="test2", actual_output="test2")
    ]
    run = EvaluationRun(
        examples=examples,
        scorers=[MockScorer()]
    )
    assert run.examples == examples

    # Test empty examples
    with pytest.raises(ValueError, match="Examples cannot be empty"):
        EvaluationRun(
            examples=[],
            scorers=[MockScorer()]
        )

    # Test mixed example types
    with pytest.raises(ValidationError):
        EvaluationRun(
            examples=[
                Example(input="test1", actual_output="test1"),
                CustomExample(input={"question": "test2"}, actual_output={"answer": "test2"})
            ],
            scorers=[MockScorer()]
        )

def test_validate_scorers():
    # Test valid scorers
    scorers = [MockScorer(), MockAPIScorer()]
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=scorers
    )
    assert run.scorers == scorers

    # Test empty scorers
    with pytest.raises(ValueError, match="Scorers cannot be empty"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[]
        )

    # Test invalid scorer type
    class InvalidScorer:
        pass

    with pytest.raises(ValidationError):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[InvalidScorer()]
        )

def test_validate_model():
    # Test valid string model
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        model="gpt-4.1"
    )
    assert run.model == "gpt-4.1"

    # Test valid list of models
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        model=["gpt-4.1", "gpt-4.1-mini"],
        aggregator="gpt-4.1"
    )
    assert run.model == ["gpt-4.1", "gpt-4.1-mini"]

    # Test valid JudgevalJudge
    judge = MockJudge()
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        model=judge
    )
    assert run.model == judge

    # Test invalid model name
    with pytest.raises(ValueError, match="Model name invalid-model not recognized"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            model="invalid-model"
        )

    # Test invalid model type
    with pytest.raises(ValidationError):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            model=123
        )

    # Test JudgevalJudge with APIJudgmentScorer
    with pytest.raises(ValueError, match="When using a judgevalJudge model, all scorers must be JudgevalScorer type"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockAPIScorer()],
            model=MockJudge()
        )

def test_validate_aggregator():
    # Test valid aggregator with list of models
    run = EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[MockScorer()],
        model=["gpt-4.1", "gpt-4.1-mini"],
        aggregator="gpt-4.1"
    )
    assert run.aggregator == "gpt-4.1"

    # Test missing aggregator with list of models
    with pytest.raises(ValueError, match="Aggregator cannot be empty"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            model=["gpt-4.1", "gpt-4.1-mini"]
        )

    # Test invalid aggregator type
    with pytest.raises(ValueError, match="Aggregator must be a string if provided"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            model=["gpt-4.1", "gpt-4.1-mini"],
            aggregator=123
        )

    # Test invalid aggregator model
    with pytest.raises(ValueError, match="Model name invalid-model not recognized"):
        EvaluationRun(
            examples=[Example(input="test", actual_output="test")],
            scorers=[MockScorer()],
            model=["gpt-4.1", "gpt-4.1-mini"],
            aggregator="invalid-model"
        )
