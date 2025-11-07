from __future__ import annotations

from typing import List, Optional

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.data.example import Example
from judgeval.v1.data.scoring_result import ScoringResult
from judgeval.v1.scorers.base_scorer import BaseScorer


class Evaluation:
    __slots__ = ("_client",)

    def __init__(self, client: JudgmentSyncClient):
        self._client = client

    def run(
        self,
        examples: List[Example],
        scorers: List[BaseScorer],
        project_name: str,
        eval_run_name: str,
        model: Optional[str] = None,
        assert_test: bool = False,
    ) -> List[ScoringResult]:
        raise NotImplementedError("Evaluation.run() not implemented in v1")
