from __future__ import annotations

from judgeval.v1.scorers.built_in.answer_correctness import AnswerCorrectnessScorer
from judgeval.v1.scorers.built_in.answer_relevancy import AnswerRelevancyScorer
from judgeval.v1.scorers.built_in.faithfulness import FaithfulnessScorer
from judgeval.v1.scorers.built_in.instruction_adherence import (
    InstructionAdherenceScorer,
)


class BuiltInScorersFactory:
    __slots__ = ()

    def answer_correctness(self, threshold: float = 0.5) -> AnswerCorrectnessScorer:
        return AnswerCorrectnessScorer.create(threshold)

    def answer_relevancy(self, threshold: float = 0.5) -> AnswerRelevancyScorer:
        return AnswerRelevancyScorer.create(threshold)

    def faithfulness(self, threshold: float = 0.5) -> FaithfulnessScorer:
        return FaithfulnessScorer.create(threshold)

    def instruction_adherence(
        self, threshold: float = 0.5
    ) -> InstructionAdherenceScorer:
        return InstructionAdherenceScorer.create(threshold)
