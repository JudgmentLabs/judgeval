from __future__ import annotations

from judgeval.v1.internal.api import JudgmentSyncClient


class ScorersFactory:
    __slots__ = "_client"

    def __init__(
        self,
        client: JudgmentSyncClient,
    ):
        self._client = client

    @property
    def prompt_scorer(self):
        from judgeval.v1.scorers.prompt_scorer.prompt_scorer_factory import (
            PromptScorerFactory,
        )

        return PromptScorerFactory(
            client=self._client,
            is_trace=False,
        )

    @property
    def trace_prompt_scorer(self):
        from judgeval.v1.scorers.prompt_scorer.prompt_scorer_factory import (
            PromptScorerFactory,
        )

        return PromptScorerFactory(
            client=self._client,
            is_trace=True,
        )

    @property
    def custom_scorer(self):
        from judgeval.v1.scorers.custom_scorer.custom_scorer_factory import (
            CustomScorerFactory,
        )

        return CustomScorerFactory()

    @property
    def built_in(self):
        from judgeval.v1.scorers.built_in.built_in_factory import BuiltInScorersFactory

        return BuiltInScorersFactory()
