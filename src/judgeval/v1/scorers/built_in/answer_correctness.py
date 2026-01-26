from __future__ import annotations

from typing import Optional

from judgeval.constants import APIScorerType
from judgeval.v1.scorers.api_scorer import APIScorer


class AnswerCorrectnessScorer(APIScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        name: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            score_type=APIScorerType.ANSWER_CORRECTNESS.value,
            required_params=["input", "actual_output", "expected_output"],
            threshold=threshold,
            name=name,
            model=model,
        )

    @staticmethod
    def create(threshold: float = 0.5) -> AnswerCorrectnessScorer:
        return AnswerCorrectnessScorer(threshold=threshold)
