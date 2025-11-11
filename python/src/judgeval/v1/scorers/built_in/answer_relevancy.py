from __future__ import annotations

from typing import Optional

from judgeval.constants import APIScorerType
from judgeval.v1.scorers.api_scorer import APIScorer


class AnswerRelevancyScorer(APIScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        name: Optional[str] = None,
        strict_mode: bool = False,
        model: Optional[str] = None,
    ):
        super().__init__(
            score_type=APIScorerType.ANSWER_RELEVANCY.value,
            required_params=["input", "actual_output"],
            threshold=threshold,
            name=name,
            strict_mode=strict_mode,
            model=model,
        )

    @staticmethod
    def create(threshold: float = 0.5) -> AnswerRelevancyScorer:
        return AnswerRelevancyScorer(threshold=threshold)
