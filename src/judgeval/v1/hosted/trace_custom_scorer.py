from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from judgeval.v1.internal.api.api_types import TraceSpan
from judgeval.v1.hosted.responses import (
    BinaryResponse,
    CategoricalResponse,
    NumericResponse,
)

R = TypeVar("R", BinaryResponse, CategoricalResponse, NumericResponse)


class TraceCustomScorer(ABC, Generic[R]):
    @abstractmethod
    async def score(self, data: List[TraceSpan]) -> R:
        pass
