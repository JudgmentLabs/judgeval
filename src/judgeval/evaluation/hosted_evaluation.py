from __future__ import annotations

from typing import List

from rich.console import Console
from rich.progress import Progress

from judgeval.data.example import Example
from judgeval.exceptions import JudgmentAPIError
from judgeval.internal.api.models import ExampleEvaluationRun
from judgeval.evaluation.evaluation_base import EvaluatorRunner


class HostedEvaluatorRunner(EvaluatorRunner[str]):
    """Hosted (server-side) scorer execution for ad-hoc evaluations.

    Hosted ad-hoc evaluation is not supported against offline-tests-v1
    servers: the queue endpoint stores results in legacy tables, while
    result polling reads the new test-run storage, so completion can
    never be observed. `_submit` fast-fails with a `JudgmentAPIError`
    (before queueing any work) directing callers to
    `client.offline_tests` for dataset-backed offline tests, or to
    local `Judge` scorers for ad-hoc evaluation.
    """

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
        raise JudgmentAPIError(
            501,
            "Hosted ad-hoc evaluation is not supported by the v1 "
            "offline-tests API: queued results land in legacy storage that "
            "the experiments endpoint no longer reads, so they can never be "
            "retrieved. Use client.offline_tests.run(...) to run "
            "dataset-backed offline tests, or pass Judge instances to "
            "evaluation.run(...) to score examples locally.",
            None,
        )
