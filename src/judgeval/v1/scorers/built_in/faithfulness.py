from __future__ import annotations

from typing import Optional

from judgeval.constants import APIScorerType
from judgeval.v1.scorers.api_scorer import APIScorer


class FaithfulnessScorer(APIScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        name: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            score_type=APIScorerType.FAITHFULNESS.value,
            required_params=["context", "actual_output"],
            threshold=threshold,
            name=name,
            model=model,
        )

    @staticmethod
    def create(threshold: float = 0.5) -> FaithfulnessScorer:
        return FaithfulnessScorer(threshold=threshold)
