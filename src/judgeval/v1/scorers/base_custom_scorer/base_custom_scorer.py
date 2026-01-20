from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from judgeval.v1.scorers.base_custom_scorer.custom_scorer_result import (
    CustomScorerResult,
)

T = TypeVar("T")


class BaseCustomScorer(ABC, Generic[T]):
    @abstractmethod
    def score(self, data: T) -> CustomScorerResult:
        """
        Produces an output score and reason for the given data.
        """
        pass
