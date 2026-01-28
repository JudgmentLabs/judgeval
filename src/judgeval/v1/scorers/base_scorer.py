from __future__ import annotations

from abc import ABC, abstractmethod

from judgeval.v1.internal.api.api_types import ScorerConfig


class BaseScorer(ABC):
    __slots__ = ()

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_scorer_config(self) -> ScorerConfig:
        pass
