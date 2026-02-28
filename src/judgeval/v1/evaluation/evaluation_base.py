from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Generic, List, Optional, TypeVar

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id
from judgeval.v1.data.example import Example
from judgeval.v1.data.scorer_data import ScorerData
from judgeval.v1.data.scoring_result import ScoringResult
from judgeval.v1.judges import Judge
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import ExampleEvaluationRun, ExperimentRunItem
from judgeval.v1.scorers.base_scorer import BaseScorer

S = TypeVar("S", Judge, BaseScorer)


class EvaluatorRunner(ABC, Generic[S]):
    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    @abstractmethod
    def _build_payload(
        self,
        eval_id: str,
        project_id: str,
        eval_run_name: str,
        created_at: str,
        examples: List[Example],
        scorers: List[S],
    ) -> ExampleEvaluationRun:
        """
        Build the ExampleEvaluationRun payload for the evaluation.
        """
        pass

    @abstractmethod
    def _submit(
        self,
        console: Console,
        project_id: str,
        eval_id: str,
        examples: List[Example],
        scorers: List[S],
        payload: ExampleEvaluationRun,
        progress: Progress,
    ) -> None:
        """
        Run the evaluation and save the results to the server.
        """
        pass

    def _poll(
        self,
        console: Console,
        project_id: str,
        eval_id: str,
        examples: List[Example],
        timeout_seconds: int,
        progress: Progress,
    ) -> tuple[list[ExperimentRunItem], str]:
        """
        Poll the server for the results of the evaluation.
        """
        task = progress.add_task("Waiting for results...", total=None)
        start_time = time.time()
        poll_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Evaluation timed out after {timeout_seconds}s")

            response = self._client.get_projects_experiments_by_run_id(
                project_id=project_id,
                run_id=eval_id,
            )
            results_data = response.get("results", []) or []
            poll_count += 1

            completed = len(results_data)
            total = len(examples)

            progress.update(
                task,
                description=f"Evals completed and saved: ({completed}/{total} completed)",
            )
            judgeval_logger.info(
                f"Poll {poll_count}: {completed}/{total} results ready"
            )

            if completed == total:
                break
            time.sleep(2)

        console.print(
            f"[green]✓[/green] Evals completed and saved in [bold]{elapsed:.1f}s[/bold]"
        )
        judgeval_logger.info(f"Evals completed and saved in {elapsed:.1f}s")

        url = (
            response.get("ui_results_url", "Failed to get UI results URL")
            or "Failed to get UI results URL"
        )
        return results_data, url

    def _display_results(
        self,
        console: Console,
        examples: List[Example],
        results_data: List[ExperimentRunItem],
        url: str,
        assert_test: bool,
    ) -> List[ScoringResult]:
        """
        Display the results of the evaluation.
        """
        console.print()
        results: List[ScoringResult] = []
        passed = 0
        failed = 0

        for i, res in enumerate(results_data):
            judgeval_logger.info(f"Processing result {i + 1}: {res.keys()}")

            scorers_raw = res.get("scorers", [])
            scorers_data: List[ScorerData] = []
            for scorer_dict in scorers_raw:
                judgeval_logger.debug(f"Scorer data fields: {scorer_dict.keys()}")

                scorers_data.append(
                    ScorerData(
                        name=scorer_dict["name"],
                        threshold=scorer_dict["threshold"],
                        success=bool(scorer_dict["success"]),
                        score=scorer_dict["score"],
                        minimum_score_range=scorer_dict.get("minimum_score_range", 0),
                        maximum_score_range=scorer_dict.get("maximum_score_range", 1),
                        reason=scorer_dict.get("reason"),
                        evaluation_model=scorer_dict.get("evaluation_model"),
                        error=scorer_dict.get("error"),
                        additional_metadata=scorer_dict.get("additional_metadata")
                        or {},
                        id=scorer_dict.get("scorer_data_id"),
                    )
                )

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
                    f"  [dim]{scorer_data.name}:[/dim] "
                    f"[{status_color}]{score_str}[/{status_color}] "
                    f"(threshold: {scorer_data.threshold})"
                )

            results.append(
                ScoringResult(
                    success=success,
                    scorers_data=scorers_data,
                    data_object=examples[i],
                )
            )

        console.print()

        if passed == len(results):
            console.print(
                f"[bold green]✓ All tests passed![/bold green] "
                f"({passed}/{len(results)})"
            )
        else:
            console.print(
                f"[bold yellow]⚠ Results:[/bold yellow] "
                f"[green]{passed} passed[/green] | "
                f"[red]{failed} failed[/red]"
            )

        console.print(f"[dim]View full details:[/dim] [link={url}]{url}[/link]\n")

        if assert_test and not all(r.success for r in results):
            raise AssertionError(
                f"Evaluation failed: {failed}/{len(results)} tests failed"
            )

        return results

    def run(
        self,
        examples: List[Example],
        scorers: List[S],
        eval_run_name: str,
        assert_test: bool = False,
        timeout_seconds: int = 300,
    ) -> List[ScoringResult]:
        project_id = expect_project_id(self._project_id)
        if project_id is None:
            return []

        console = Console()
        eval_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        console.print("\n[bold cyan]Starting Evaluation[/bold cyan]")
        console.print(f"[dim]Run:[/dim] {eval_run_name}")
        console.print(f"[dim]Project:[/dim] {self._project_name}")
        console.print(
            f"[dim]Examples:[/dim] {len(examples)} | [dim]Scorers:[/dim] {len(scorers)}"
        )

        judgeval_logger.info(f"Starting evaluation: {eval_run_name}")
        judgeval_logger.info(f"Examples: {len(examples)}, Scorers: {len(scorers)}")

        payload = self._build_payload(
            eval_id, project_id, eval_run_name, created_at, examples, scorers
        )

        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            self._submit(
                console,
                project_id,
                eval_id,
                examples,
                scorers,
                payload,
                progress,
            )
            results_data, url = self._poll(
                console,
                project_id,
                eval_id,
                examples,
                timeout_seconds,
                progress,
            )

        return self._display_results(console, examples, results_data, url, assert_test)
