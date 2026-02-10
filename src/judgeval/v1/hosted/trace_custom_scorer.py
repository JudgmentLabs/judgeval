from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from judgeval.v1.data.trace import Trace
from judgeval.v1.hosted.responses import (
    BinaryResponse,
    CategoricalResponse,
    NumericResponse,
)

R = TypeVar("R", BinaryResponse, CategoricalResponse, NumericResponse)


class TraceCustomScorer(ABC, Generic[R]):
    @abstractmethod
    async def score(self, data: Trace) -> R:
        pass
