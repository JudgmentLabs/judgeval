from __future__ import annotations

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.evaluation.evaluation import Evaluation


class EvaluationFactory:
    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: str,
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    def create(self) -> Evaluation:
        return Evaluation(
            client=self._client,
            project_id=self._project_id,
            project_name=self._project_name,
        )
