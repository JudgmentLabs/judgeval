from __future__ import annotations

from typing import Optional

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.evaluation.evaluation import Evaluation
from judgeval.v1.evaluation.noop_evaluation import NoopEvaluation
from judgeval.utils.guards import expect_project_id


class EvaluationFactory:
    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    def create(self) -> Evaluation:
        project_id = expect_project_id(self._project_id, context="evaluation creation")
        if not project_id:
            return NoopEvaluation(project_name=self._project_name)

        return Evaluation(
            client=self._client,
            project_id=project_id,
            project_name=self._project_name,
        )
