from __future__ import annotations

from typing import Optional

from judgeval.v1.scorers.api_scorer import APIScorer


class InstructionAdherenceScorer(APIScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        name: Optional[str] = None,
        strict_mode: bool = False,
        model: Optional[str] = None,
    ):
        super().__init__(
            score_type="instruction_adherence",
            required_params=["input", "actual_output"],
            threshold=threshold,
            name=name,
            strict_mode=strict_mode,
            model=model,
        )

    @staticmethod
    def create(threshold: float = 0.5) -> InstructionAdherenceScorer:
        return InstructionAdherenceScorer(threshold=threshold)
