from __future__ import annotations

from typing import List, Optional, Union

from judgeval.logger import judgeval_logger
from judgeval.v1.data.example import Example
from judgeval.v1.data.scoring_result import ScoringResult
from judgeval.v1.judges import Judge
from judgeval.v1.hosted.example_custom_scorer import ExampleCustomScorer
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.scorers.base_scorer import BaseScorer
from judgeval.v1.evaluation.local_evaluation import LocalEvaluatorRunner
from judgeval.v1.evaluation.hosted_evaluation import HostedEvaluatorRunner


class Evaluation:
    __slots__ = ("_local", "_hosted")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._local = LocalEvaluatorRunner(client, project_id, project_name)
        self._hosted = HostedEvaluatorRunner(client, project_id, project_name)

    def run(
        self,
        examples: List[Example],
        scorers: Union[List[BaseScorer], List[Judge | ExampleCustomScorer]],
        eval_run_name: str,
        assert_test: bool = False,
        timeout_seconds: int = 300,
    ) -> List[ScoringResult]:
        local_scorers = [
            s for s in scorers if isinstance(s, (Judge, ExampleCustomScorer))
        ]
        hosted_scorers = [s for s in scorers if isinstance(s, (BaseScorer))]
        if len(local_scorers) > 0 and len(hosted_scorers) > 0:
            judgeval_logger.error(
                "Running both local and hosted scorers is not supported. Please run your evaluation with either local or hosted scorers, but not both."
            )
            return []
        if len(local_scorers) == 0 and len(hosted_scorers) == 0:
            judgeval_logger.error(
                "No valid local or hosted scorers provided. Please provide at least one local or hosted scorer."
            )
            return []
        if any(isinstance(s, ExampleCustomScorer) for s in local_scorers):
            judgeval_logger.warning(
                "ExampleCustomScorer is deprecated. Please use Judge instead."
            )
        if len(local_scorers) > 0:
            return self._local.run(
                examples,
                local_scorers,
                eval_run_name,
                assert_test=assert_test,
                timeout_seconds=timeout_seconds,
            )
        if len(hosted_scorers) > 0:
            return self._hosted.run(
                examples,
                hosted_scorers,
                eval_run_name,
                assert_test=assert_test,
                timeout_seconds=timeout_seconds,
            )
        return []
