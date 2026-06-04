from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id
from judgeval.data.example import Example
from judgeval.data.scorer_data import ScorerData
from judgeval.data.scoring_result import ScoringResult
from judgeval.internal.api import JudgmentSyncClient
from judgeval.internal.api.models import ExampleEvaluationRun, ExperimentRunItem
from judgeval.judges import Judge

S = TypeVar("S", str, Judge)


def _binary_label(value: bool) -> str:
    return "Yes" if value else "No"


def _scorer_value(scorer_dict: dict[str, Any]) -> str | float | None:
    score_type = scorer_dict.get("score_type")

    if score_type == "binary":
        bool_value = scorer_dict.get("bool_value")
        if isinstance(bool_value, bool):
            return _binary_label(bool_value)
        value = scorer_dict.get("value")
        if isinstance(value, bool):
            return _binary_label(value)
        if isinstance(value, str):
            return value
        str_value = scorer_dict.get("str_value")
        if isinstance(str_value, str) and str_value:
            return str_value

    if score_type == "categorical":
        str_value = scorer_dict.get("str_value")
        if isinstance(str_value, str):
            return str_value
        value = scorer_dict.get("value")
        if isinstance(value, str):
            return value

    bool_value = scorer_dict.get("bool_value")
    if isinstance(bool_value, bool):
        return _binary_label(bool_value)

    str_value = scorer_dict.get("str_value")
    if isinstance(str_value, str) and str_value:
        return str_value

    num_value = scorer_dict.get("num_value")
    if isinstance(num_value, (int, float)):
        return float(num_value)

    value = scorer_dict.get("value")
    if isinstance(value, bool):
        return _binary_label(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)

    score = scorer_dict.get("score")
    if isinstance(score, (int, float)):
        return float(score)

    return None


class EvaluatorRunner(ABC, Generic[S]):
    """Abstract base for evaluation runners.

    Concrete implementations handle either hosted (server-side) or local
    (in-process) scorer execution.  The generic parameter ``S`` is ``str``
    for hosted scorers or ``Judge`` for local scorers.
    """

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
    ) -> int:
        """
        Run the evaluation and save the results to the server.
        Returns the number of unique examples to expect results for.
        """
        pass

    def _poll(
        self,
        console: Console,
        project_id: str,
        eval_id: str,
        expected_count: int,
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

            progress.update(
                task,
                description=f"Evals completed and saved: ({completed}/{expected_count} completed)",
            )
            judgeval_logger.info(
                f"Poll {poll_count}: {completed}/{expected_count} results ready"
            )

            if completed == expected_count:
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

        if assert_test:
            judgeval_logger.warning(
                "assert_test is deprecated and ignored by the current "
                "evaluation result payload."
            )

        for i, res in enumerate(results_data):
            judgeval_logger.info(f"Processing result {i + 1}: {res.keys()}")

            scorers_raw = res.get("scorers", [])
            scorers_data: List[ScorerData] = []
            for scorer_dict in scorers_raw:
                judgeval_logger.debug(f"Scorer data fields: {scorer_dict.keys()}")

                scorers_data.append(
                    ScorerData(
                        name=scorer_dict["name"],
                        value=_scorer_value(scorer_dict),
                        score_type=scorer_dict.get("score_type"),
                        minimum_score_range=scorer_dict.get("minimum_score_range", 0),
                        maximum_score_range=scorer_dict.get("maximum_score_range", 1),
                        evaluation_model=scorer_dict.get("evaluation_model"),
                        error=scorer_dict.get("error"),
                        additional_metadata=scorer_dict.get("additional_metadata")
                        or {},
                        id=scorer_dict.get("scorer_data_id"),
                    )
                )

            console.print(f"[cyan]•[/cyan] Example {i + 1}:")

            for scorer_data in scorers_data:
                value = scorer_data.value
                value_str = f"{value:.3f}" if isinstance(value, float) else value
                if value_str is None:
                    value_str = "N/A"
                console.print(
                    f"  [dim]{scorer_data.name}:[/dim] [cyan]{value_str}[/cyan]"
                )
                if scorer_data.error:
                    console.print(f"    [red]{scorer_data.error}[/red]")

            results.append(
                ScoringResult(
                    scorers_data=scorers_data,
                    data_object=examples[i],
                )
            )

        console.print()
        console.print(f"[bold green]✓[/bold green] Results ready ({len(results)})")

        console.print(f"[dim]View full details:[/dim] [link={url}]{url}[/link]\n")

        return results

    def run(
        self,
        examples: List[Example],
        scorers: List[S],
        eval_run_name: str,
        assert_test: bool = False,
        timeout_seconds: int = 300,
    ) -> List[ScoringResult]:
        """Execute an evaluation run and return results.

        Args:
            examples: Examples to evaluate.
            scorers: Scorers to run (strings or ``Judge`` instances).
            eval_run_name: Name for this evaluation run.
            assert_test: Deprecated and ignored by the current evaluation
                result payload.
            timeout_seconds: Maximum time to wait for results.

        Returns:
            A list of ``ScoringResult`` objects, one per example.
        """
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
            expected_count = self._submit(
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
                expected_count,
                timeout_seconds,
                progress,
            )

        return self._display_results(console, examples, results_data, url, assert_test)
