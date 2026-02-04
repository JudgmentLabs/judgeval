from __future__ import annotations

from typing import TYPE_CHECKING

from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer

if TYPE_CHECKING:
    from judgeval.v1.internal.api.api_types import (
        BaseScorer as BaseScorerDict,
        ScorerConfig,
    )


class NoopCustomScorer(CustomScorer):
    """A no-op CustomScorer that indicates the scorer could not be loaded.

    Used when project_id is not available or scorer doesn't exist,
    allowing code to continue without raising exceptions.
    """

    __slots__ = ()

    def __init__(self, name: str = ""):
        self._name = name
        self._project_id = ""

    def get_scorer_config(self) -> ScorerConfig:
        return {
            "score_type": "noop",
            "name": self._name,
            "threshold": 0.0,
        }

    def to_dict(self) -> BaseScorerDict:
        return {"score_type": "noop", "name": self._name}
