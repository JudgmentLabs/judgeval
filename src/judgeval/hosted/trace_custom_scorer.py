import warnings
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from judgeval.data.trace import Trace
from judgeval.hosted.responses import (
    BinaryResponse,
    CategoricalResponse,
    NumericResponse,
)

R = TypeVar("R", BinaryResponse, CategoricalResponse, NumericResponse)


class TraceCustomScorer(ABC, Generic[R]):
    """**Deprecated.** Use `Judge[R]` from `judgeval.judges` instead."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from TraceCustomScorer which is deprecated. "
            "Use Judge[R] from judgeval.judges instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    async def score(self, data: Trace) -> R:
        pass
