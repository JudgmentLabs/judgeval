from __future__ import annotations

from typing import Optional

from judgeval.v1.internal.api import JudgmentSyncClient


class ScorersFactory:
    __slots__ = ("_client", "_default_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        default_project_id: Optional[str] = None,
        project_name: Optional[str] = None,
    ):
        self._client = client
        self._default_project_id = default_project_id
        self._project_name = project_name

    @property
    def prompt_scorer(self):
        from judgeval.v1.scorers.prompt_scorer.prompt_scorer_factory import (
            PromptScorerFactory,
        )

        return PromptScorerFactory(
            client=self._client,
            is_trace=False,
            default_project_id=self._default_project_id,
        )

    @property
    def trace_prompt_scorer(self):
        from judgeval.v1.scorers.prompt_scorer.prompt_scorer_factory import (
            PromptScorerFactory,
        )

        return PromptScorerFactory(
            client=self._client,
            is_trace=True,
            default_project_id=self._default_project_id,
        )

    @property
    def custom_scorer(self):
        from judgeval.v1.scorers.custom_scorer.custom_scorer_factory import (
            CustomScorerFactory,
        )

        return CustomScorerFactory(
            client=self._client,
            default_project_id=self._default_project_id,
        )

    @property
    def built_in(self):
        from judgeval.v1.scorers.built_in.built_in_factory import BuiltInScorersFactory

        return BuiltInScorersFactory()
