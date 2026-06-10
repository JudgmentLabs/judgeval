from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import orjson
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from judgeval.data.example import Example
from judgeval.data.scorer_data import ScorerData
from judgeval.data.scoring_result import ScoringResult
from judgeval.exceptions import (
    JudgmentAPIError,
    JudgmentTestError,
    map_judgment_api_error,
)
from judgeval.evaluation.evaluation_base import _scorer_value
from judgeval.internal.api import JudgmentSyncClient
from judgeval.logger import judgeval_logger
from judgeval.offline_tests.types import OfflineTestResult, TestConfig

AgentFunction = Callable[..., Any]
PassConditionFn = Callable[[Dict[str, Any], List[ScorerData]], bool]

TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})


def normalize_judge_versions(
    judge_versions: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Validate and normalize `judge_versions` entries.

    Each entry must identify a judge by `name` (or `judge_id`) and may pin
    a `tag`, `version`, or `major_version`/`minor_version` pair. Judges
    not listed default to their `prod` tag (else latest) server-side.

    Raises:
        ValueError: If an entry is not a dict or identifies no judge.
    """
    if not judge_versions:
        return None

    allowed_keys = (
        "judge_id",
        "name",
        "tag",
        "version",
        "major_version",
        "minor_version",
    )
    normalized: List[Dict[str, Any]] = []
    for entry in judge_versions:
        if not isinstance(entry, dict):
            raise ValueError(
                "judge_versions entries must be dicts like "
                '{"name": "my-judge", "tag": "prod"}'
            )
        if not entry.get("name") and not entry.get("judge_id"):
            raise ValueError(
                "judge_versions entries require a 'name' (or 'judge_id') key"
            )
        normalized.append(
            {k: entry[k] for k in allowed_keys if entry.get(k) is not None}
        )
    return normalized


def has_duplicate_judges(judge_versions: List[Dict[str, Any]]) -> bool:
    """Whether two `judge_versions` entries reference the same judge."""
    seen = set()
    for entry in judge_versions:
        key = entry.get("judge_id") or entry.get("name")
        if key in seen:
            return True
        seen.add(key)
    return False


def build_agent_kwargs(
    agent_function: AgentFunction, data: Dict[str, Any]
) -> Dict[str, Any]:
    """Map an example's data fields onto the agent entrypoint's parameters.

    Every data field is passed as a same-named keyword argument. Raises if
    the entrypoint cannot accept a field or requires a parameter the
    example does not provide.

    Raises:
        TypeError: On any mismatch between the entrypoint signature and
            the example's fields.
    """
    signature = inspect.signature(agent_function)
    params = signature.parameters

    accepts_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    keyword_params = {
        name: p
        for name, p in params.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    if not accepts_var_keyword:
        unexpected = sorted(set(data) - set(keyword_params))
        if unexpected:
            raise TypeError(
                f"Agent entrypoint {agent_function.__name__}() does not accept "
                f"example field(s) {unexpected}. Add matching parameters "
                "(or **kwargs) to the entrypoint, or update the dataset schema."
            )

    missing = sorted(
        name
        for name, p in keyword_params.items()
        if p.default is inspect.Parameter.empty and name not in data
    )
    if missing:
        raise TypeError(
            f"Agent entrypoint {agent_function.__name__}() requires parameter(s) "
            f"{missing} that are not present in the example data."
        )

    return dict(data)


def _parse_reason(raw: Any) -> Dict[str, Any]:
    """Coerce a stored scorer reason into the `{text, citations?}` wire shape."""
    if isinstance(raw, dict) and isinstance(raw.get("text"), str):
        return raw
    if isinstance(raw, str):
        try:
            parsed = orjson.loads(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                return parsed
        except orjson.JSONDecodeError:
            pass
        return {"text": raw}
    return {"text": ""}


def _parse_metadata(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = orjson.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except orjson.JSONDecodeError:
            return None
    return None


def _reason_text(raw: Any) -> Optional[str]:
    reason = _parse_reason(raw)
    text = reason.get("text")
    return text if text else None


class OfflineTestRunner:
    """Executes the offline-test lifecycle for a test config.

    Used by `client.offline_tests.run()` -- you don't instantiate it
    directly. The lifecycle is:

    1. `POST test-runs` -- creates the run and queues server-side judge
       evaluation over the dataset examples.
    2. Optionally call the agent entrypoint once per dataset example,
       wrapped in an `OfflineTracer` so each call produces an offline
       trace.
    3. Wait for the run to reach a terminal status and fetch per-example
       results.
    4. Evaluate the optional `pass_condition_fn` per row and report the
       enriched results back (echoing `evaluation_run_id`, `success`, and
       `agent_offline_trace_id`).
    """

    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: str,
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    # ------------------------------------------------------------------ #
    #  Lifecycle steps                                                   #
    # ------------------------------------------------------------------ #

    def create_test_run(
        self,
        test_config: TestConfig,
        dataset_version: Optional[int | str] = None,
        judge_versions: Optional[List[Dict[str, Any]]] = None,
        versioned_results: Optional[bool] = None,
        source: str = "sdk",
    ) -> Dict[str, Any]:
        """Create (prepare + queue) a test run and return the prepared payload."""
        payload: Dict[str, Any] = {
            "test_config_id": test_config.id,
            "source": source,
        }
        if isinstance(dataset_version, int):
            payload["dataset_version_number"] = dataset_version
        elif isinstance(dataset_version, str):
            payload["dataset_version_id"] = dataset_version

        normalized = normalize_judge_versions(judge_versions)
        if normalized:
            payload["judge_versions"] = normalized
            if versioned_results is None and has_duplicate_judges(normalized):
                versioned_results = True
        if versioned_results is not None:
            payload["versioned_results"] = versioned_results

        try:
            prepared = self._client.post_projects_test_runs(
                project_id=self._project_id,
                payload=payload,  # type: ignore[arg-type]
            )
            return dict(prepared)
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e,
                f"Failed to create test run for config '{test_config.name}': {e.detail}",
            ) from e

    def run_agent(
        self,
        agent_function: AgentFunction,
        examples: List[Dict[str, Any]],
        progress: Optional[Progress] = None,
    ) -> Dict[str, str]:
        """Call the agent entrypoint once per dataset example.

        Each call is wrapped in the `OfflineTracer` machinery so it
        produces a dedicated offline trace; the resulting trace IDs are
        returned keyed by example ID. Entrypoint/example field mismatches
        raise immediately; runtime errors inside the agent are recorded on
        the trace and logged, and the loop continues. The previously
        active tracer (if any) is restored once the loop finishes, so
        subsequent `@observe` spans do not route to the offline endpoint.
        """
        from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
        from judgeval.trace.offline_tracer import OfflineTracer

        proxy = JudgmentTracerProvider.get_instance()
        previous_tracer = proxy.get_active_tracer()

        captured: List[Example] = []
        tracer = OfflineTracer.create(
            project_name=self._project_name,
            api_key=self._client.api_key,
            organization_id=self._client.organization_id,
            api_url=self._client.base_url,
            set_active=True,
            dataset=captured,
        )
        try:
            wrapped = tracer.observe(agent_function, span_type="agent")
            is_async = inspect.iscoroutinefunction(agent_function)

            task = None
            if progress is not None:
                task = progress.add_task(
                    f"Running agent over {len(examples)} example(s)...", total=None
                )

            agent_traces: Dict[str, str] = {}
            for index, example in enumerate(examples):
                example_id = example.get("example_id") or ""
                data = example.get("data") or {}
                kwargs = build_agent_kwargs(agent_function, data)

                before = len(captured)
                try:
                    if is_async:
                        asyncio.run(wrapped(**kwargs))
                    else:
                        wrapped(**kwargs)
                except Exception as exc:
                    judgeval_logger.error(
                        f"Agent entrypoint raised for example {example_id}: {exc}"
                    )

                for produced in captured[before:]:
                    offline_trace_id = produced._properties.get("offline_trace_id")
                    if example_id and offline_trace_id:
                        agent_traces[example_id] = offline_trace_id
                        break

                if progress is not None and task is not None:
                    progress.update(
                        task,
                        description=f"Running agent... ({index + 1}/{len(examples)})",
                    )
        finally:
            tracer.force_flush()
            proxy.restore_active(previous_tracer)
            proxy.deregister(tracer)
            tracer._tracer_provider.shutdown()

        return agent_traces

    def wait_for_completion(
        self,
        test_run_id: str,
        timeout_seconds: int,
        progress: Optional[Progress] = None,
    ) -> str:
        """Poll the test run until it reaches a terminal status."""
        task = None
        if progress is not None:
            task = progress.add_task("Waiting for judge results...", total=None)

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Test run {test_run_id} did not complete within {timeout_seconds}s"
                )

            response = self._client.get_projects_test_runs_by_test_run_id(
                project_id=self._project_id,
                test_run_id=test_run_id,
            )
            status = str((response.get("test_run") or {}).get("status") or "")
            if progress is not None and task is not None:
                progress.update(
                    task,
                    description=f"Waiting for judge results... (status: {status})",
                )
            if status in TERMINAL_STATUSES:
                return status
            time.sleep(2)

    def fetch_items(self, test_run_id: str) -> Tuple[List[Dict[str, Any]], str]:
        """Fetch per-example scorer rows for a test run."""
        response = self._client.get_projects_test_runs_by_test_run_id_items(
            project_id=self._project_id,
            test_run_id=test_run_id,
        )
        return (response.get("results") or [], response.get("ui_results_url") or "")

    def build_results(
        self,
        items: List[Dict[str, Any]],
        agent_traces: Dict[str, str],
        pass_condition_fn: Optional[PassConditionFn] = None,
    ) -> List[ScoringResult]:
        """Convert raw test-run items into `ScoringResult` objects.

        When `pass_condition_fn` is provided it is called once per row
        with `(data_fields, scorer_data_list)` and its boolean outcome is
        recorded on every `ScorerData` of the row.
        """
        results: List[ScoringResult] = []
        for item in items:
            example_id = item.get("example_id") or ""
            data = item.get("data") or {}

            example = Example(example_id=example_id)
            entry = item.get("example")
            if isinstance(entry, dict) and entry.get("created_at"):
                example.created_at = entry["created_at"]
            for key, value in data.items():
                example._properties[key] = value

            scorers_data: List[ScorerData] = []
            for scorer in item.get("scorers") or []:
                metadata = {
                    "judge_id": scorer.get("judge_id"),
                    "judge_major_version": scorer.get("judge_major_version"),
                    "judge_minor_version": scorer.get("judge_minor_version"),
                }
                reason_text = _reason_text(scorer.get("reason"))
                if reason_text:
                    metadata["reason"] = reason_text
                scorers_data.append(
                    ScorerData(
                        name=scorer.get("judge_name") or "",
                        value=_scorer_value(scorer),
                        score_type=scorer.get("score_type"),
                        error=scorer.get("error"),
                        additional_metadata=metadata,
                        success=scorer.get("success"),
                    )
                )

            if pass_condition_fn is not None:
                passed = bool(pass_condition_fn(dict(data), scorers_data))
                for scorer_data in scorers_data:
                    scorer_data.success = passed

            results.append(
                ScoringResult(
                    scorers_data=scorers_data,
                    data_object=example,
                    trace_id=agent_traces.get(example_id),
                )
            )
        return results

    def report_results(
        self,
        test_run_id: str,
        prepared: Dict[str, Any],
        items: List[Dict[str, Any]],
        results: List[ScoringResult],
        agent_traces: Dict[str, str],
    ) -> None:
        """Report enriched results back to the server.

        Echoes each scorer row with its `evaluation_run_id` (for exact
        attribution), the `success` outcome of the pass condition, and
        the example's `agent_offline_trace_id`. The rows replace the
        server-side rows for this run/example/judge-version.
        """
        refs_by_version: Dict[Tuple[str, str, int, int], str] = {}
        refs_by_name: Dict[Tuple[str, str], str] = {}
        for ref in prepared.get("evaluation_runs") or []:
            run_id = ref.get("run_id")
            if not run_id:
                continue
            refs_by_version[
                (
                    ref.get("example_id") or "",
                    ref.get("judge_id") or "",
                    int(ref.get("judge_major_version") or 0),
                    int(ref.get("judge_minor_version") or 0),
                )
            ] = run_id
            refs_by_name[(ref.get("example_id") or "", ref.get("judge_name") or "")] = (
                run_id
            )

        results_by_example = {
            result.data_object.example_id: result
            for result in results
            if isinstance(result.data_object, Example)
        }

        wire_results: List[Dict[str, Any]] = []
        for item in items:
            example_id = item.get("example_id") or ""
            data = item.get("data") or {}
            result = results_by_example.get(example_id)
            success_by_index: List[Optional[bool]] = (
                [scorer.success for scorer in result.scorers_data] if result else []
            )

            scorers_data: List[Dict[str, Any]] = []
            for index, scorer in enumerate(item.get("scorers") or []):
                score_type = scorer.get("score_type")
                entry: Dict[str, Any] = {
                    "scorer_name": scorer.get("judge_name") or "",
                    "score_type": score_type,
                    "reason": _parse_reason(scorer.get("reason")),
                }
                if score_type == "binary":
                    entry["bool_value"] = bool(scorer.get("bool_value"))
                elif score_type == "categorical":
                    entry["str_value"] = str(scorer.get("str_value") or "")
                elif score_type == "numeric":
                    entry["num_value"] = float(scorer.get("num_value") or 0)
                else:
                    judgeval_logger.warning(
                        f"Skipping scorer row with unknown score_type {score_type!r}"
                    )
                    continue

                metadata = _parse_metadata(scorer.get("metadata"))
                if metadata is not None:
                    entry["metadata"] = metadata
                if scorer.get("error") is not None:
                    entry["error"] = scorer.get("error")

                evaluation_run_id = refs_by_version.get(
                    (
                        example_id,
                        scorer.get("judge_id") or "",
                        int(scorer.get("judge_major_version") or 0),
                        int(scorer.get("judge_minor_version") or 0),
                    )
                ) or refs_by_name.get((example_id, scorer.get("judge_name") or ""))
                if evaluation_run_id:
                    entry["evaluation_run_id"] = evaluation_run_id

                if index < len(success_by_index):
                    entry["success"] = success_by_index[index]
                scorers_data.append(entry)

            if not scorers_data:
                continue

            data_object: Dict[str, Any] = {
                "example_id": example_id,
                "created_at": item.get("created_at") or "",
            }
            entry_example = item.get("example")
            if isinstance(entry_example, dict):
                if entry_example.get("created_at"):
                    data_object["created_at"] = entry_example["created_at"]
                if entry_example.get("offline_trace_id"):
                    data_object["offline_trace_id"] = entry_example["offline_trace_id"]
            data_object.update(data)
            agent_offline_trace_id = agent_traces.get(example_id)
            if agent_offline_trace_id:
                data_object["agent_offline_trace_id"] = agent_offline_trace_id

            wire_results.append(
                {"scorers_data": scorers_data, "data_object": data_object}
            )

        if not wire_results:
            return

        run_payload: Dict[str, Any] = {
            "id": test_run_id,
            "project_id": self._project_id,
            "eval_name": f"Offline Test Run {test_run_id}",
            "examples": [wire["data_object"] for wire in wire_results],
            "custom_scorers": [],
            "judgment_scorers": [],
        }

        try:
            self._client.post_projects_eval_results(
                project_id=self._project_id,
                payload={
                    "test_run_id": test_run_id,
                    "results": wire_results,  # type: ignore[typeddict-item]
                    "run": run_payload,  # type: ignore[typeddict-item]
                },
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e, f"Failed to report test run results: {e.detail}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Orchestration                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        test_config: TestConfig,
        agent_function: Optional[AgentFunction] = None,
        judge_versions: Optional[List[Dict[str, Any]]] = None,
        dataset_version: Optional[int | str] = None,
        versioned_results: Optional[bool] = None,
        pass_condition_fn: Optional[PassConditionFn] = None,
        assert_test: bool = False,
        timeout_seconds: int = 600,
    ) -> OfflineTestResult:
        """Execute the full offline-test lifecycle for a test config."""
        if assert_test and pass_condition_fn is None:
            raise ValueError(
                "assert_test=True requires a pass_condition_fn to decide "
                "whether each row passes."
            )

        console = Console()
        console.print("\n[bold cyan]Starting Offline Test[/bold cyan]")
        console.print(f"[dim]Config:[/dim] {test_config.name}")
        console.print(f"[dim]Project:[/dim] {self._project_name}")

        prepared = self.create_test_run(
            test_config,
            dataset_version=dataset_version,
            judge_versions=judge_versions,
            versioned_results=versioned_results,
        )
        test_run = prepared.get("test_run") or {}
        test_run_id = test_run.get("id") or ""
        examples = prepared.get("examples") or []
        ui_results_url = prepared.get("ui_results_url") or ""

        console.print(
            f"[dim]Run:[/dim] {test_run_id} | [dim]Examples:[/dim] {len(examples)}"
        )
        judgeval_logger.info(
            f"Created test run {test_run_id} with {len(examples)} examples"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            agent_traces: Dict[str, str] = {}
            if agent_function is not None:
                agent_traces = self.run_agent(agent_function, examples, progress)

            status = self.wait_for_completion(test_run_id, timeout_seconds, progress)

        items, items_url = self.fetch_items(test_run_id)
        ui_results_url = items_url or ui_results_url

        results = self.build_results(items, agent_traces, pass_condition_fn)

        if pass_condition_fn is not None or agent_traces:
            self.report_results(test_run_id, prepared, items, results, agent_traces)

        self._display_results(console, status, results, ui_results_url)

        outcome = OfflineTestResult(
            test_run_id=test_run_id,
            status=status,
            ui_results_url=ui_results_url,
            results=results,
            agent_offline_trace_ids=agent_traces,
        )

        if assert_test:
            self._assert_all_passed(outcome)
        return outcome

    def _assert_all_passed(self, outcome: OfflineTestResult) -> None:
        if outcome.status != "completed":
            raise JudgmentTestError(
                f"Test run {outcome.test_run_id} finished with status "
                f"'{outcome.status}'"
            )
        failed = [
            result.data_object.example_id
            for result in outcome.results
            if isinstance(result.data_object, Example)
            and any(scorer.success is False for scorer in result.scorers_data)
        ]
        if failed or outcome.passed is not True:
            raise JudgmentTestError(
                f"Test run {outcome.test_run_id} failed its pass condition for "
                f"{len(failed)} example(s): {failed}"
            )

    def _display_results(
        self,
        console: Console,
        status: str,
        results: List[ScoringResult],
        ui_results_url: str,
    ) -> None:
        console.print()
        for i, result in enumerate(results):
            console.print(f"[cyan]•[/cyan] Example {i + 1}:")
            for scorer_data in result.scorers_data:
                value = scorer_data.value
                value_str = f"{value:.3f}" if isinstance(value, float) else value
                if value_str is None:
                    value_str = "N/A"
                suffix = ""
                if scorer_data.success is True:
                    suffix = " [green](passed)[/green]"
                elif scorer_data.success is False:
                    suffix = " [red](failed)[/red]"
                console.print(
                    f"  [dim]{scorer_data.name}:[/dim] [cyan]{value_str}[/cyan]{suffix}"
                )
                if scorer_data.error:
                    console.print(f"    [red]{scorer_data.error}[/red]")

        console.print()
        status_color = "green" if status == "completed" else "red"
        console.print(
            f"[bold {status_color}]✓[/bold {status_color}] Test run {status} "
            f"({len(results)} result(s))"
        )
        if ui_results_url:
            console.print(
                f"[dim]View full details:[/dim] "
                f"[link={ui_results_url}]{ui_results_url}[/link]\n"
            )
