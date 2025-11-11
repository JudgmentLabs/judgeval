from __future__ import annotations

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.evaluation.evaluation import Evaluation


class EvaluationFactory:
    __slots__ = "_client"

    def __init__(
        self,
        client: JudgmentSyncClient,
    ):
        self._client = client

    def create(self) -> Evaluation:
        return Evaluation(client=self._client)
