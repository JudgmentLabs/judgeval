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
    @abstractmethod
    async def score(self, data: Example) -> R:
        pass
