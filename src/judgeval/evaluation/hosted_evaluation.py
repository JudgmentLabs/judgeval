from __future__ import annotations

from typing import List

from rich.console import Console
from rich.progress import Progress

from judgeval.logger import judgeval_logger
from judgeval.data.example import Example
from judgeval.internal.api.models import ExampleEvaluationRun
from judgeval.evaluation.evaluation_base import EvaluatorRunner


class HostedEvaluatorRunner(EvaluatorRunner[str]):
    def _build_payload(
        self,
        eval_id: str,
        project_id: str,
        eval_run_name: str,
        created_at: str,
        examples: List[Example],
        scorers: List[str],
    ) -> ExampleEvaluationRun:
        return {
            "id": eval_id,
            "project_id": project_id,
            "eval_name": eval_run_name,
            "created_at": created_at,
            "examples": [e.to_dict() for e in examples],
            "judgment_scorers": [{"name": name} for name in scorers],
            "custom_scorers": [],
        }

    def _submit(
        self,
        console: Console,
        project_id: str,
        eval_id: str,
        examples: List[Example],
        scorers: List[str],
        payload: ExampleEvaluationRun,
        progress: Progress,
    ) -> int:
        task = progress.add_task("Submitting evaluation...", total=None)
        self._client.post_projects_eval_queue_examples(
            project_id=project_id,
            payload=payload,
        )
        judgeval_logger.info(f"Evaluation submitted: {eval_id}")
        progress.update(task, description="Running evaluation...")
        return len(examples)
