from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from judgeval.v1.data.example import Example
from judgeval.v1.hosted.responses import (
    BaseResponse,
)

R = TypeVar("R", bound=BaseResponse)


class Judge(ABC, Generic[R]):
    @abstractmethod
    async def score(self, data: Example) -> R:
        pass
