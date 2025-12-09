from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import ExampleEvaluationRun
from judgeval.v1.data.example import Example
from judgeval.v1.data.scoring_result import ScoringResult
from judgeval.v1.data.scorer_data import ScorerData
from judgeval.v1.scorers.base_scorer import BaseScorer
from judgeval.logger import judgeval_logger


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
        timeout_seconds: int = 300,
    ) -> List[ScoringResult]:
        console = Console()
        eval_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        console.print("\n[bold cyan]Starting Evaluation[/bold cyan]")
        console.print(f"[dim]Run:[/dim] {eval_run_name}")
        console.print(f"[dim]Project:[/dim] {project_name}")
        console.print(
            f"[dim]Examples:[/dim] {len(examples)} | [dim]Scorers:[/dim] {len(scorers)}"
        )
        if model:
            console.print(f"[dim]Model:[/dim] {model}")

        judgeval_logger.info(f"Starting evaluation: {eval_run_name}")
        judgeval_logger.info(f"Examples: {len(examples)}, Scorers: {len(scorers)}")

        payload: ExampleEvaluationRun = {
            "id": eval_id,
            "project_name": project_name,
            "eval_name": eval_run_name,
            "created_at": created_at,
            "examples": [e.to_dict() for e in examples],
            "judgment_scorers": [s.get_scorer_config() for s in scorers],
            "custom_scorers": [],
        }

        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Submitting evaluation...", total=None)
            self._client.add_to_run_eval_queue_examples(payload)
            judgeval_logger.info(f"Evaluation submitted: {eval_id}")

            progress.update(task, description="Running evaluation...")
            start_time = time.time()
            poll_count = 0

            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Evaluation timed out after {timeout_seconds}s")

                response = self._client.fetch_experiment_run(
                    {"experiment_run_id": eval_id, "project_name": project_name}
                )
                results_data = response.get("results", []) or []
                poll_count += 1

                completed = len(results_data)
                total = len(examples)
                progress.update(
                    task,
                    description=f"Running evaluation... ({completed}/{total} completed)",
                )
                judgeval_logger.info(
                    f"Poll {poll_count}: {completed}/{total} results ready"
                )

                if completed == total:
                    break
                time.sleep(2)

        console.print(
            f"[green]✓[/green] Evaluation completed in [bold]{elapsed:.1f}s[/bold]"
        )
        judgeval_logger.info(f"Evaluation completed in {elapsed:.1f}s")

        console.print()
        results = []
        passed = 0
        failed = 0

        for i, res in enumerate(results_data):
            judgeval_logger.info(f"Processing result {i + 1}: {res.keys()}")

            scorers_raw = res.get("scorers", [])
            scorers_data = []
            for scorer_dict in scorers_raw:
                judgeval_logger.debug(f"Scorer data fields: {scorer_dict.keys()}")

                scorer_fields = {
                    "name": scorer_dict.get("name"),
                    "threshold": scorer_dict.get("threshold"),
                    "success": scorer_dict.get("success"),
                    "score": scorer_dict.get("score"),
                    "minimum_score_range": scorer_dict.get("minimum_score_range", 0),
                    "maximum_score_range": scorer_dict.get("maximum_score_range", 1),
                    "reason": scorer_dict.get("reason"),
                    "strict_mode": scorer_dict.get("strict_mode"),
                    "evaluation_model": scorer_dict.get("evaluation_model"),
                    "error": scorer_dict.get("error"),
                    "additional_metadata": scorer_dict.get("additional_metadata", {}),
                    "id": scorer_dict.get("scorer_data_id") or scorer_dict.get("id"),
                }
                scorers_data.append(ScorerData(**scorer_fields))

            success = all(s.success for s in scorers_data)

            if success:
                passed += 1
                console.print(
                    f"[green]✓[/green] Example {i + 1}: [green]PASSED[/green]"
                )
            else:
                failed += 1
                console.print(f"[red]✗[/red] Example {i + 1}: [red]FAILED[/red]")

            for scorer_data in scorers_data:
                score_str = (
                    f"{scorer_data.score:.3f}"
                    if scorer_data.score is not None
                    else "N/A"
                )
                status_color = "green" if scorer_data.success else "red"
                console.print(
                    f"  [dim]{scorer_data.name}:[/dim] [{status_color}]{score_str}[/{status_color}] (threshold: {scorer_data.threshold})"
                )

            results.append(
                ScoringResult(
                    success=success,
                    scorers_data=scorers_data,
                )
            )

        console.print()
        url = response.get("ui_results_url", "")

        if passed == len(results):
            console.print(
                f"[bold green]✓ All tests passed![/bold green] ({passed}/{len(results)})"
            )
        else:
            console.print(
                f"[bold yellow]⚠ Results:[/bold yellow] [green]{passed} passed[/green] | [red]{failed} failed[/red]"
            )

        console.print(f"[dim]View full details:[/dim] [link={url}]{url}[/link]\n")

        if assert_test and not all(r.success for r in results):
            raise AssertionError(
                f"Evaluation failed: {failed}/{len(results)} tests failed"
            )

        return results
