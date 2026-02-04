from __future__ import annotations

from typing import List, Optional

from judgeval.v1.data.example import Example
from judgeval.v1.data.scoring_result import ScoringResult
from judgeval.v1.scorers.base_scorer import BaseScorer
from judgeval.v1.evaluation.evaluation import Evaluation


class NoopEvaluation(Evaluation):
    """A no-op Evaluation that silently skips all operations.

    Used when project_id is not available, allowing code to continue
    without raising exceptions.
    """

    __slots__ = ()

    def __init__(self, project_name: str = ""):
        super().__init__(
            client=None,  # type: ignore[arg-type]
            project_id="",
            project_name=project_name,
        )

    def _validate_scorer_project(self, scorer: BaseScorer) -> None:
        pass

    def run(
        self,
        examples: List[Example],
        scorers: List[BaseScorer],
        eval_run_name: str,
        model: Optional[str] = None,
        assert_test: bool = False,
        timeout_seconds: int = 300,
    ) -> List[ScoringResult]:
        return []
