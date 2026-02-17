from __future__ import annotations

import asyncio
import concurrent.futures
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from judgeval.utils.guards import expect_project_id
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import (
    ExampleEvaluationRun,
    LogEvalResultsRequest,
)
from judgeval.v1.data.example import Example
from judgeval.v1.data.scoring_result import ScoringResult
from judgeval.v1.data.scorer_data import ScorerData
from judgeval.v1.scorers.base_scorer import BaseScorer
from judgeval.v1.hosted.example_custom_scorer import ExampleCustomScorer
from judgeval.v1.hosted.responses import (
    BinaryResponse,
    CategoricalResponse,
    NumericResponse,
    ScorerResponse,
)
from judgeval.logger import judgeval_logger


def _safe_run_async(coro):
    """Run an async coroutine from sync code, handling an already-running event loop."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


def _response_to_scorer_data(
    scorer_name: str,
    response: ScorerResponse,
    threshold: float = 0.5,
) -> ScorerData:
    """Convert a BinaryResponse/CategoricalResponse/NumericResponse to ScorerData."""
    additional_metadata: Dict[str, Any]
    if isinstance(response, BinaryResponse):
        score = 1.0 if response.value else 0.0
        minimum_score_range = 0.0
        maximum_score_range = 1.0
        additional_metadata = {}
    elif isinstance(response, NumericResponse):
        score = float(response.value)
        minimum_score_range = 0.0
        maximum_score_range = 1.0
        additional_metadata = {}
    elif isinstance(response, CategoricalResponse):
        score = None
        minimum_score_range = 0.0
        maximum_score_range = 1.0
        additional_metadata = {"category": response.value}
    else:
        raise TypeError(f"Unexpected response type: {type(response)}")

    return ScorerData(
        name=scorer_name,
        threshold=threshold,
        success=score >= threshold if score is not None else True,
        score=score,
        minimum_score_range=minimum_score_range,
        maximum_score_range=maximum_score_range,
        reason=response.reason,
        additional_metadata=additional_metadata,
    )


def _response_to_api_scorer(
    scorer_name: str,
    response: ScorerResponse,
) -> Dict[str, Any]:
    """Build the API-format scorer payload matching the backend schema."""
    reason: Dict[str, Any] = {"text": response.reason}
    if response.citations:
        reason["citations"] = [c.model_dump() for c in response.citations]

    base: Dict[str, Any] = {
        "scorer_name": scorer_name,
        "reason": reason,
        "error": None,
    }

    if isinstance(response, BinaryResponse):
        base["score_type"] = "binary"
        base["bool_value"] = bool(response.value)
    elif isinstance(response, CategoricalResponse):
        base["score_type"] = "categorical"
        base["str_value"] = str(response.value)
    elif isinstance(response, NumericResponse):
        base["score_type"] = "numeric"
        base["num_value"] = float(response.value)
    else:
        raise TypeError(f"Unexpected response type: {type(response)}")

    return base


async def _run_custom_scorers(
    examples: List[Example],
    scorers: List[ExampleCustomScorer],
    on_example_complete: Optional[Callable[[int], None]] = None,
) -> List[List[Tuple[ScorerData, ScorerResponse]]]:
    """Run all custom scorers across all examples concurrently.

    Returns a list (per example) of (ScorerData, raw response) pairs.
    """

    async def score_example(i: int, example: Example) -> tuple:
        responses = await asyncio.gather(*[scorer.score(example) for scorer in scorers])
        pairs = [
            (_response_to_scorer_data(type(scorer).__name__, response), response)
            for scorer, response in zip(scorers, responses)
        ]
        if on_example_complete:
            on_example_complete(i)
        return i, pairs

    completed = await asyncio.gather(
        *[score_example(i, ex) for i, ex in enumerate(examples)]
    )
    completed_sorted = sorted(completed, key=lambda x: x[0])
    return [pairs for _, pairs in completed_sorted]


class Evaluation:
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

    def _validate_scorer_project(self, scorer: BaseScorer) -> None:
        scorer_project_id = getattr(scorer, "_project_id", None)
        if scorer_project_id is not None and scorer_project_id != self._project_id:
            judgeval_logger.warning(
                f"Rejecting scorer '{scorer.get_name()}' with different project_id: "
                f"{scorer_project_id} != {self._project_id}"
            )
            raise ValueError(
                f"Scorer '{scorer.get_name()}' belongs to project '{scorer_project_id}', "
                f"but this evaluation is bound to project '{self._project_name}' ({self._project_id})"
            )

    def _print_header(
        self,
        console: Console,
        eval_run_name: str,
        examples: List[Example],
        scorers: Union[List[BaseScorer], List[ExampleCustomScorer]],
        model: Optional[str] = None,
    ) -> None:
        console.print("\n[bold cyan]Starting Evaluation[/bold cyan]")
        console.print(f"[dim]Run:[/dim] {eval_run_name}")
        console.print(f"[dim]Project:[/dim] {self._project_name}")
        console.print(
            f"[dim]Examples:[/dim] {len(examples)} | [dim]Scorers:[/dim] {len(scorers)}"
        )
        if model:
            console.print(f"[dim]Model:[/dim] {model}")

    def _assemble_results(
        self,
        console: Console,
        examples: List[Example],
        all_scorers_data: List[List[ScorerData]],
    ) -> tuple:
        """Shared result assembly and printing for both hosted and local paths.

        Returns (results, passed, failed).
        """
        results: List[ScoringResult] = []
        passed = 0
        failed = 0

        for i, scorers_data in enumerate(all_scorers_data):
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
                    data_object=examples[i],
                )
            )

        return results, passed, failed

    def _print_summary(
        self,
        console: Console,
        results: List[ScoringResult],
        passed: int,
        failed: int,
        url: str = "",
        assert_test: bool = False,
    ) -> List[ScoringResult]:
        console.print()
        if passed == len(results):
            console.print(
                f"[bold green]✓ All tests passed![/bold green] ({passed}/{len(results)})"
            )
        else:
            console.print(
                f"[bold yellow]⚠ Results:[/bold yellow] [green]{passed} passed[/green] | [red]{failed} failed[/red]"
            )

        if url:
            console.print(f"[dim]View full details:[/dim] [link={url}]{url}[/link]\n")

        if assert_test and not all(r.success for r in results):
            raise AssertionError(
                f"Evaluation failed: {failed}/{len(results)} tests failed"
            )

        return results

    def _run_local(
        self,
        examples: List[Example],
        scorers: List[ExampleCustomScorer],
        eval_run_name: str,
        project_id: str,
        model: Optional[str] = None,
        assert_test: bool = False,
    ) -> List[ScoringResult]:
        """Run custom scorers locally and log results to the API."""
        console = Console()
        eval_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        self._print_header(console, eval_run_name, examples, scorers, model)
        judgeval_logger.info(f"Starting local evaluation: {eval_run_name}")
        judgeval_logger.info(f"Examples: {len(examples)}, Scorers: {len(scorers)}")

        console.print()
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Running local scorers... (0/{len(examples)} completed)",
                total=None,
            )
            completed_count = 0

            def on_example_complete(i: int) -> None:
                nonlocal completed_count
                completed_count += 1
                progress.update(
                    task,
                    description=f"Running local scorers... ({completed_count}/{len(examples)} completed)",
                )

            all_pairs = _safe_run_async(
                _run_custom_scorers(examples, scorers, on_example_complete)
            )

        elapsed = time.time() - start_time
        console.print(
            f"[green]✓[/green] Evaluation completed in [bold]{elapsed:.1f}s[/bold]"
        )
        judgeval_logger.info(f"Local evaluation completed in {elapsed:.1f}s")

        all_scorers_data = [[sd for sd, _ in pairs] for pairs in all_pairs]

        console.print()
        results, passed, failed = self._assemble_results(
            console, examples, all_scorers_data
        )

        url = ""
        try:
            api_results = []
            for i, pairs in enumerate(all_pairs):
                api_scorers = [
                    _response_to_api_scorer(type(scorers[j]).__name__, resp)
                    for j, (_, resp) in enumerate(pairs)
                ]
                api_results.append(
                    {
                        "scorers_data": api_scorers,
                        "data_object": examples[i].to_dict(),
                    }
                )

            payload: LogEvalResultsRequest = {
                # TODO: fix ScoringResult autogen type in api_types.py
                "results": api_results,  # type: ignore[typeddict-item]
                "run": {
                    "id": eval_id,
                    "project_id": project_id,
                    "eval_name": eval_run_name,
                    "created_at": created_at,
                    "examples": [e.to_dict() for e in examples],
                    "judgment_scorers": [],
                    "custom_scorers": [],
                },
            }
            response = self._client.post_projects_eval_results(
                project_id=project_id,
                payload=payload,
            )
            url = response.get("ui_results_url", "")
        except Exception as e:
            judgeval_logger.error(f"Failed to log local eval results: {e}")

        return self._print_summary(console, results, passed, failed, url, assert_test)

    def _run_hosted(
        self,
        examples: List[Example],
        scorers: List[BaseScorer],
        eval_run_name: str,
        project_id: str,
        model: Optional[str] = None,
        assert_test: bool = False,
        timeout_seconds: int = 300,
    ) -> List[ScoringResult]:
        """Submit scorers to the hosted queue and poll for results."""
        for scorer in scorers:
            self._validate_scorer_project(scorer)

        console = Console()
        eval_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        self._print_header(console, eval_run_name, examples, scorers, model)
        judgeval_logger.info(f"Starting evaluation: {eval_run_name}")
        judgeval_logger.info(f"Examples: {len(examples)}, Scorers: {len(scorers)}")

        payload: ExampleEvaluationRun = {
            "id": eval_id,
            "project_id": project_id,
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
            self._client.post_projects_eval_queue_examples(
                project_id=project_id,
                payload=payload,
            )
            judgeval_logger.info(f"Evaluation submitted: {eval_id}")

            progress.update(task, description="Running evaluation...")
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
        all_scorers_data: List[List[ScorerData]] = []
        for i, res in enumerate(results_data):
            judgeval_logger.info(f"Processing result {i + 1}: {res.keys()}")

            scorers_raw = res.get("scorers", [])
            scorers_data = []
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
            all_scorers_data.append(scorers_data)

        results, passed, failed = self._assemble_results(
            console, examples, all_scorers_data
        )

        url = response.get("ui_results_url", "") or ""

        return self._print_summary(console, results, passed, failed, url, assert_test)

    def run(
        self,
        examples: List[Example],
        scorers: Union[List[BaseScorer], List[ExampleCustomScorer]],
        eval_run_name: str,
        model: Optional[str] = None,
        assert_test: bool = False,
        timeout_seconds: int = 300,
    ) -> List[ScoringResult]:
        project_id = expect_project_id(self._project_id)
        if project_id is None:
            return []

        if all(isinstance(scorer, ExampleCustomScorer) for scorer in scorers):
            return self._run_local(
                examples=examples,
                scorers=scorers,  # type: ignore
                eval_run_name=eval_run_name,
                project_id=project_id,
                model=model,
                assert_test=assert_test,
            )

        return self._run_hosted(
            examples=examples,
            scorers=scorers,  # type: ignore
            eval_run_name=eval_run_name,
            project_id=project_id,
            model=model,
            assert_test=assert_test,
            timeout_seconds=timeout_seconds,
        )
