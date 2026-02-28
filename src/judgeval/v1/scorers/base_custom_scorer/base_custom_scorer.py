import warnings
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from judgeval.v1.scorers.base_custom_scorer.custom_scorer_result import (
    CustomScorerResult,
)

T = TypeVar("T")


class BaseCustomScorer(ABC, Generic[T]):
    """Deprecated: Use ``Judge[R]`` from ``judgeval.v1.judges`` instead."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from BaseCustomScorer which is deprecated. "
            "Use Judge[R] from judgeval.v1.judges instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    def score(self, data: T) -> CustomScorerResult:
        """
        Produces an output score and reason for the given data.
        """
        pass
