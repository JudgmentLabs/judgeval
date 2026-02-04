from __future__ import annotations

from typing import TYPE_CHECKING

from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer
from judgeval.v1.internal.api.api_types import ScorerConfig

if TYPE_CHECKING:
    from judgeval.v1.internal.api.api_types import BaseScorer as BaseScorerDict


class NoopCustomScorer(CustomScorer):
    """A no-op CustomScorer that indicates the scorer could not be loaded.

    Used when project_id is not available or scorer doesn't exist,
    allowing code to continue without raising exceptions.
    """

    __slots__ = ()

    def __init__(self, name: str = ""):
        super().__init__(name=name, project_id="")

    def get_scorer_config(self) -> ScorerConfig:
        return ScorerConfig(
            score_type="noop",
            name=self._name,
            threshold=0.0,
        )

    def to_dict(self) -> BaseScorerDict:
        return {"score_type": "noop", "name": self._name}
