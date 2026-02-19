from __future__ import annotations

from typing import List

from rich.console import Console
from rich.progress import Progress

from judgeval.logger import judgeval_logger
from judgeval.v1.data.example import Example
from judgeval.v1.internal.api.api_types import ExampleEvaluationRun
from judgeval.v1.scorers.base_scorer import BaseScorer
from judgeval.v1.evaluation.evaluation_base import EvaluatorRunner


class HostedEvaluatorRunner(EvaluatorRunner[BaseScorer]):
    def _validate_scorer_project(self, scorer: BaseScorer) -> None:
        scorer_project_id = getattr(scorer, "_project_id", None)
        if scorer_project_id is not None and scorer_project_id != self._project_id:
            judgeval_logger.warning(
                f"Rejecting scorer '{scorer.get_name()}' with different project_id: "
                f"{scorer_project_id} != {self._project_id}"
            )
            raise ValueError(
                f"Scorer '{scorer.get_name()}' belongs to project "
                f"'{scorer_project_id}', but this evaluation is bound to project "
                f"'{self._project_name}' ({self._project_id})"
            )

    def _build_payload(
        self,
        eval_id: str,
        project_id: str,
        eval_run_name: str,
        created_at: str,
        examples: List[Example],
        scorers: List[BaseScorer],
    ) -> ExampleEvaluationRun:
        return {
            "id": eval_id,
            "project_id": project_id,
            "eval_name": eval_run_name,
            "created_at": created_at,
            "examples": [e.to_dict() for e in examples],
            "judgment_scorers": [s.get_scorer_config() for s in scorers],
            "custom_scorers": [],
        }

    def _submit(
        self,
        console: Console,
        project_id: str,
        eval_id: str,
        examples: List[Example],
        scorers: List[BaseScorer],
        payload: ExampleEvaluationRun,
        progress: Progress,
    ) -> None:
        for scorer in scorers:
            self._validate_scorer_project(scorer)

        task = progress.add_task("Submitting evaluation...", total=None)
        self._client.post_projects_eval_queue_examples(
            project_id=project_id,
            payload=payload,
        )
        judgeval_logger.info(f"Evaluation submitted: {eval_id}")
        progress.update(task, description="Running evaluation...")
