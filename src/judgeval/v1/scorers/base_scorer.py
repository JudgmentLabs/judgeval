from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from judgeval.v1.internal.api.api_types import ScorerConfig


class BaseScorer(ABC):
    __slots__ = ()

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_scorer_config(self) -> ScorerConfig:
        pass

    def get_project_id(self) -> Optional[str]:
        """Return the project_id this scorer belongs to, or None if not project-specific."""
        return None
