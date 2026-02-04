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
    allowing code to continue without raising exceptions. Logging happens
    once at factory level, not on every method call (consistent with
    legacy NoOpJudgmentSpanProcessor).
    """

    __slots__ = ()

    def __init__(self, name: str = ""):
        # Don't call super().__init__ to avoid requiring project_id
        self._name = name
        self._project_id = ""

    def get_scorer_config(self) -> ScorerConfig:
        # Return safe config instead of raising NotImplementedError
        # to support graceful degradation if accidentally used
        return {
            "score_type": "noop",
            "name": self._name,
            "threshold": 0.0,
        }

    def to_dict(self) -> BaseScorerDict:
        return {"score_type": "noop", "name": self._name}
