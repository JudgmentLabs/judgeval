from __future__ import annotations

from judgeval.v1.internal.api.api_types import ScorerConfig
from judgeval.v1.scorers.prompt_scorer.prompt_scorer import PromptScorer


class NoopPromptScorer(PromptScorer):
    """A no-op PromptScorer that indicates the scorer could not be loaded.

    Used when project_id is not available or scorer doesn't exist,
    allowing code to continue without raising exceptions. Logging happens
    once at factory level, not on every method call (consistent with
    legacy NoOpJudgmentSpanProcessor).
    """

    __slots__ = ()

    def __init__(self, name: str = ""):
        super().__init__(
            name=name,
            prompt="",
            threshold=0.0,
        )

    def get_scorer_config(self) -> ScorerConfig:
        return ScorerConfig(
            score_type=self._score_type,
            threshold=0.0,
            name=self._name,
            kwargs={"prompt": ""},
        )
