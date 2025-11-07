from __future__ import annotations

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.scorers.built_in.built_in_factory import BuiltInScorersFactory
from judgeval.v1.scorers.custom_scorer.custom_scorer_factory import CustomScorerFactory
from judgeval.v1.scorers.prompt_scorer.prompt_scorer_factory import PromptScorerFactory


class ScorersFactory:
    __slots__ = "_client"

    def __init__(
        self,
        client: JudgmentSyncClient,
    ):
        self._client = client

    def prompt_scorer(self) -> PromptScorerFactory:
        return PromptScorerFactory(
            client=self._client,
            is_trace=False,
        )

    def trace_prompt_scorer(self) -> PromptScorerFactory:
        return PromptScorerFactory(
            client=self._client,
            is_trace=True,
        )

    def custom_scorer(self) -> CustomScorerFactory:
        return CustomScorerFactory()

    def built_in(self) -> BuiltInScorersFactory:
        return BuiltInScorersFactory()
