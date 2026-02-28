import warnings
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from judgeval.v1.data.example import Example
from judgeval.v1.hosted.responses import (
    BinaryResponse,
    CategoricalResponse,
    NumericResponse,
)

R = TypeVar("R", BinaryResponse, CategoricalResponse, NumericResponse)


class ExampleCustomScorer(ABC, Generic[R]):
    """Deprecated: Use ``Judge[R]`` from ``judgeval.v1.judges`` instead."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from ExampleCustomScorer which is deprecated. "
            "Use Judge[R] from judgeval.v1.judges instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    async def score(self, data: Example) -> R:
        pass
