from __future__ import annotations

from typing import Optional

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.evaluation.evaluation import Evaluation


class EvaluationFactory:
    __slots__ = ("_client", "_default_project_id")

    def __init__(
        self,
        client: JudgmentSyncClient,
        default_project_id: Optional[str] = None,
    ):
        self._client = client
        self._default_project_id = default_project_id

    def create(self) -> Evaluation:
        return Evaluation(
            client=self._client,
            default_project_id=self._default_project_id,
        )
