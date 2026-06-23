from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, cast

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
from judgeval.utils.url import url_for

AgentFunction = Callable[..., Any]
PassConditionFn = Callable[[Dict[str, Any], List[ScorerData]], bool]


class JudgeVersionPin(TypedDict, total=False):
    """A single ``judge_versions`` entry.

    Identify the judge by ``name`` or ``judge_id``; optionally pin a
    ``tag``, a ``version`` string, or a ``major_version``/``minor_version``
    pair. All keys are optional so callers can mix identification and
    pinning styles, but every entry must carry a ``name`` or ``judge_id``
    (enforced at runtime by ``normalize_judge_versions``).
    """

    name: str
    judge_id: str
    tag: str
    version: str
    major_version: int
    minor_version: int


TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})
EXAMPLES_PAGE_SIZE = 100
# Server default and maximum page size for test-run items reads.
ITEMS_PAGE_SIZE = 200


def normalize_judge_versions(
    judge_versions: Optional[List[JudgeVersionPin]],
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

    allowed_keys = frozenset(
        {
            "judge_id",
            "name",
            "tag",
            "version",
            "major_version",
            "minor_version",
        }
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
        # Iterate items() rather than index by a variable key: keeps the
        # entry typed as JudgeVersionPin (TypedDict allows .items(); it only
        # rejects dynamic entry[key] access) while dropping unset/None keys.
        normalized.append(
            {k: v for k, v in entry.items() if k in allowed_keys and v is not None}
        )
    return normalized


def build_agent_kwargs(
    agent_function: AgentFunction,
    data: Dict[str, Any],
    field_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Map an example's data fields onto the agent entrypoint's parameters.

    Each declared parameter is filled from the example field of the same name,
    or from a custom-mapped field via ``field_mapping`` (``{param_name:
    dataset_field_name}``). Example fields the entrypoint does not declare are
    ignored -- so a dataset can carry extra columns (e.g. ``trace``) the agent
    doesn't use -- unless the entrypoint accepts ``**kwargs``, in which case the
    leftover fields are passed through too. The match succeeds as long as the
    example supplies every required (no-default) parameter.

    Raises:
        TypeError: only if a required parameter has no matching example field.
    """
    field_mapping = field_mapping or {}
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

    kwargs: Dict[str, Any] = {}
    missing: List[str] = []
    for name, p in keyword_params.items():
        source = field_mapping.get(name, name)
        if source in data:
            kwargs[name] = data[source]
        elif p.default is inspect.Parameter.empty:
            missing.append(source)

    if missing:
        raise TypeError(
            f"Agent entrypoint {agent_function.__name__}() requires example "
            f"field(s) {sorted(missing)} that are not present in the example "
            "data (check the dataset schema or pass a field_mapping)."
        )

    # A **kwargs entrypoint opts into receiving everything; forward any example
    # fields not already bound to a named parameter under their own names.
    if accepts_var_keyword:
        consumed = {field_mapping.get(name, name) for name in keyword_params}
        for field, value in data.items():
            if field not in consumed and field not in kwargs:
                kwargs[field] = value

    return kwargs


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


def _reason_text(raw: Any) -> Optional[str]:
    reason = _parse_reason(raw)
    text = reason.get("text")
    return text if text else None


class OfflineTestRunner:
    """Executes the offline-test lifecycle for a test config.

    Used by `client.offline_tests.run()` -- you don't instantiate it
    directly. The lifecycle is:

    1. Resolve the dataset version under test and fetch its examples.
    2. Optionally call the agent entrypoint once per dataset example,
       wrapped in an `OfflineTracer` so each call produces an offline
       trace. All traces are flushed before the run is created.
    3. `POST test-runs` -- creates the run pinned to the dataset version
       fetched in step 1, attaching any agent traces (`agent_traces`) so
       server-side judge evaluation is queued with the agent's trace in
       judge context.
    4. Wait for the run to reach a terminal status and fetch per-example
       results.
    5. Evaluate the optional `pass_condition_fn` per row and PATCH the
       per-evaluation-run `success` outcomes onto the test run. Skipped
       entirely when no pass condition is given.
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

    def resolve_dataset_version(
        self,
        test_config: TestConfig,
        dataset_version: Optional[int | str] = None,
    ) -> Dict[str, Any]:
        """Resolve the dataset version a run will evaluate.

        Returns the raw version entry (`version_id`, `version_number`,
        ...) for the requested version -- a version number (int), a
        version ID (str), or the latest version when `dataset_version`
        is None. The resolved version is the one whose examples are
        fetched *and* the one pinned at run creation, so the two always
        match.

        Raises:
            ValueError: If the dataset has no versions or no version
                matches `dataset_version`.
        """
        try:
            response = (
                self._client.get_projects_datasets_by_dataset_identifier_versions(
                    project_id=self._project_id,
                    dataset_identifier=test_config.dataset_id,
                )
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e,
                f"Failed to fetch versions for dataset of test config "
                f"'{test_config.name}': {e.detail}",
            ) from e

        versions = [v for v in response.get("versions") or [] if isinstance(v, dict)]
        if dataset_version is None:
            if not versions:
                raise ValueError(
                    f"Dataset of test config '{test_config.name}' has no versions"
                )
            return max(versions, key=lambda v: int(v.get("version_number") or 0))

        if isinstance(dataset_version, int):
            for version in versions:
                if int(version.get("version_number") or 0) == dataset_version:
                    return version
        else:
            for version in versions:
                if version.get("version_id") == dataset_version:
                    return version
        raise ValueError(
            f"Dataset version {dataset_version!r} does not exist for the "
            f"dataset of test config '{test_config.name}'"
        )

    def fetch_examples(
        self,
        test_config: TestConfig,
        version_number: int,
    ) -> List[Dict[str, Any]]:
        """Fetch every example of one dataset version.

        Pages through the dataset's ``/page`` endpoint which returns:

        .. code-block:: json

            {
              "dataset": {...},
              "entries": [
                {
                  "item":    {"id": "...", "version_added": 1, "example_id": "...", "created_at": "<item ts>", ...},
                  "example": {"example_id": "...", "data": {...}, "offline_trace_id": "...|null", "metadata": {...}, "created_at": "<example ts>"}
                }
              ],
              "metadata": {"hasMore": true|false, "nextCursor": {"created_at": "...", "example_id": "..."} | null}
            }

        Returns a list of dicts with keys ``example_id``, ``data`` (always a
        ``dict``), ``offline_trace_id``, and ``created_at`` drawn from the
        nested ``example`` object.  Pagination uses ``metadata.nextCursor``
        verbatim; the cursor is never derived from individual entries.
        """
        examples: List[Dict[str, Any]] = []
        cursor_created_at: Optional[str] = None
        cursor_example_id: Optional[str] = None
        page_count = 0
        max_pages = 10_000

        while True:
            if page_count >= max_pages:
                raise RuntimeError(
                    f"fetch_examples exceeded {max_pages} pages for dataset of "
                    f"test config '{test_config.name}'; aborting to prevent runaway loop"
                )
            try:
                page = self._client.get_projects_datasets_by_dataset_identifier_page(
                    project_id=self._project_id,
                    dataset_identifier=test_config.dataset_id,
                    version=str(version_number),
                    limit=str(EXAMPLES_PAGE_SIZE),
                    cursor_created_at=cursor_created_at,
                    cursor_example_id=cursor_example_id,
                )
            except JudgmentAPIError as e:
                raise map_judgment_api_error(
                    e,
                    f"Failed to fetch examples for dataset of test config "
                    f"'{test_config.name}': {e.detail}",
                ) from e
            page_count += 1

            entries = [e for e in page.get("entries") or [] if isinstance(e, dict)]
            for entry in entries:
                example = entry["example"]
                data = example.get("data")
                if isinstance(data, str):
                    try:
                        data = orjson.loads(data)
                    except orjson.JSONDecodeError:
                        data = None
                examples.append(
                    {
                        "example_id": example.get("example_id"),
                        "data": data if isinstance(data, dict) else {},
                        "offline_trace_id": example.get("offline_trace_id"),
                        "created_at": example.get("created_at"),
                    }
                )

            metadata = page.get("metadata") or {}
            has_more = metadata.get("hasMore")
            next_cursor = metadata.get("nextCursor")
            if not has_more or not next_cursor:
                break

            cursor_created_at = str(next_cursor["created_at"])
            cursor_example_id = str(next_cursor["example_id"])

        return examples

    def create_test_run(
        self,
        test_config: TestConfig,
        dataset_version: Optional[int | str] = None,
        judge_versions: Optional[List[JudgeVersionPin]] = None,
        source: str = "sdk",
        agent_traces: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a test run and return the prepared payload.

        Creation queues server-side judge evaluation immediately. When
        `agent_traces` is provided, each
        `{example_id, agent_offline_trace_id}` pair is attached so judges
        evaluate with the agent's trace in context; the server validates
        the example IDs against the dataset version (422 on unknown or
        duplicate IDs). Callers must therefore flush agent traces before
        calling this.
        """
        payload: Dict[str, Any] = {
            "test_config_id": test_config.id,
            "source": source,
        }
        if name:
            payload["name"] = name
        if isinstance(dataset_version, int):
            payload["dataset_version_number"] = dataset_version
        elif isinstance(dataset_version, str):
            payload["dataset_version_id"] = dataset_version

        normalized = normalize_judge_versions(judge_versions)
        if normalized:
            payload["judge_versions"] = normalized

        if agent_traces:
            payload["agent_traces"] = [
                {"example_id": example_id, "agent_offline_trace_id": trace_id}
                for example_id, trace_id in agent_traces.items()
            ]

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
        field_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Call the agent entrypoint once per dataset example.

        Each call is wrapped in the `OfflineTracer` machinery so it
        produces a dedicated offline trace; the resulting trace IDs are
        returned keyed by example ID. Entrypoint/example field mismatches
        raise immediately; runtime errors inside the agent are recorded on
        the trace and logged, and the loop continues. The previously
        active tracer (if any) is restored once the loop finishes, so
        subsequent `@observe` spans do not route to the offline endpoint.

        Before returning, the offline tracer is force-flushed and its
        provider shut down, so every agent trace is exported by the time
        the test run is created with these trace IDs attached.
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
                kwargs = build_agent_kwargs(agent_function, data, field_mapping)

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
        """Fetch every per-example scorer row for a test run.

        Pages through ``GET .../test-runs/{id}/items`` with ``limit`` /
        ``cursor`` query params (keyset-paginated by ``example_id``; the
        server caps pages at 200 rows) until ``has_more`` is false or
        ``next_cursor`` is null. Older servers that do not paginate omit
        both fields and are treated as a single full page.

        The server truncates each scorer's ``str_value``, ``reason``, and
        ``error`` to 128 characters in these rows; follow
        ``ui_results_url`` for the full values.

        Uses the raw ``JudgmentSyncClient._request`` helper (same httpx
        machinery, headers, and error handling) because the generated
        client does not expose the ``limit``/``cursor`` query params yet.
        A future client regen will pick them up.
        """
        items: List[Dict[str, Any]] = []
        ui_results_url = ""
        cursor: Optional[str] = None
        page_count = 0
        max_pages = 10_000

        while True:
            if page_count >= max_pages:
                raise RuntimeError(
                    f"fetch_items exceeded {max_pages} pages for test run "
                    f"{test_run_id}; aborting to prevent runaway loop"
                )
            params: Dict[str, Any] = {"limit": ITEMS_PAGE_SIZE}
            if cursor is not None:
                params["cursor"] = cursor
            response = cast(
                Dict[str, Any],
                self._client._request(
                    "GET",
                    url_for(
                        f"/v1/projects/{self._project_id}/test-runs/{test_run_id}/items",
                        self._client.base_url,
                    ),
                    params,
                ),
            )
            page_count += 1

            items.extend(response.get("results") or [])
            if not ui_results_url:
                ui_results_url = response.get("ui_results_url") or ""

            next_cursor = response.get("next_cursor")
            if not response.get("has_more") or not next_cursor:
                break
            cursor = str(next_cursor)

        return items, ui_results_url

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

    def report_success(
        self,
        test_run_id: str,
        prepared: Dict[str, Any],
        items: List[Dict[str, Any]],
        results: List[ScoringResult],
    ) -> None:
        """PATCH per-row pass-condition outcomes onto the test run.

        Sends one `{evaluation_run_id, success}` entry per scorer row to
        `PATCH .../test-runs/{id}/success`. Each row's `evaluation_run_id`
        comes from the prepare response refs (matched by judge version,
        falling back to judge name); the server validates the IDs and
        re-inserts its own stored rows with the new success -- nothing
        else is echoed back.
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

        successes: List[Dict[str, Any]] = []
        for item in items:
            example_id = item.get("example_id") or ""
            result = results_by_example.get(example_id)
            success_by_index: List[Optional[bool]] = (
                [scorer.success for scorer in result.scorers_data] if result else []
            )

            for index, scorer in enumerate(item.get("scorers") or []):
                evaluation_run_id = refs_by_version.get(
                    (
                        example_id,
                        scorer.get("judge_id") or "",
                        int(scorer.get("judge_major_version") or 0),
                        int(scorer.get("judge_minor_version") or 0),
                    )
                ) or refs_by_name.get((example_id, scorer.get("judge_name") or ""))
                if not evaluation_run_id:
                    judgeval_logger.warning(
                        f"No evaluation run ref for scorer "
                        f"{scorer.get('judge_name')!r} of example {example_id!r}; "
                        "skipping its success update"
                    )
                    continue
                successes.append(
                    {
                        "evaluation_run_id": evaluation_run_id,
                        "success": success_by_index[index]
                        if index < len(success_by_index)
                        else None,
                    }
                )

        if not successes:
            return

        try:
            self._patch_test_run_success(test_run_id, successes)
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e, f"Failed to report test run successes: {e.detail}"
            ) from e

    def _patch_test_run_success(
        self, test_run_id: str, successes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call `PATCH .../test-runs/{id}/success` via the raw client.

        The route is not in the generated client yet, so this reuses
        `JudgmentSyncClient._request` (same httpx machinery, headers, and
        error handling). A future client regen will pick up the route.
        """
        return cast(
            Dict[str, Any],
            self._client._request(
                "PATCH",
                url_for(
                    f"/v1/projects/{self._project_id}/test-runs/{test_run_id}/success",
                    self._client.base_url,
                ),
                payload={"successes": successes},
            ),
        )

    # ------------------------------------------------------------------ #
    #  Orchestration                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        test_config: TestConfig,
        agent_function: Optional[AgentFunction] = None,
        judge_versions: Optional[List[JudgeVersionPin]] = None,
        dataset_version: Optional[int | str] = None,
        pass_condition_fn: Optional[PassConditionFn] = None,
        assert_test: bool = False,
        timeout_seconds: int = 600,
        run_name: Optional[str] = None,
        field_mapping: Optional[Dict[str, str]] = None,
    ) -> OfflineTestResult:
        """Execute the full offline-test lifecycle for a test config.

        When ``agent_function`` is omitted, no agent is invoked: the judges
        score each example's existing trace (the dataset's trace-typed
        column / ``offline_trace_id``). When provided, the agent is run once
        per example first and the judges score the resulting agent trace.
        """
        if assert_test and pass_condition_fn is None:
            raise ValueError(
                "assert_test=True requires a pass_condition_fn to decide "
                "whether each row passes."
            )

        console = Console()
        console.print("\n[bold cyan]Starting Offline Test[/bold cyan]")
        console.print(f"[dim]Config:[/dim] {test_config.name}")
        console.print(f"[dim]Project:[/dim] {self._project_name}")

        version = self.resolve_dataset_version(test_config, dataset_version)
        version_number = int(version.get("version_number") or 0)
        examples = self.fetch_examples(test_config, version_number)

        console.print(
            f"[dim]Dataset version:[/dim] {version_number} | "
            f"[dim]Examples:[/dim] {len(examples)}"
        )

        # Pin the run to the exact version the examples were fetched from.
        pinned_version: int | str = (
            dataset_version if isinstance(dataset_version, str) else version_number
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # The agent runs before the test run exists; run_agent flushes
            # the offline tracer (and shuts down its provider) before
            # returning, so all agent traces are exported by the time they
            # are attached to the run below.
            agent_traces: Dict[str, str] = {}
            if agent_function is not None and examples:
                agent_traces = self.run_agent(
                    agent_function, examples, progress, field_mapping
                )

            prepared = self.create_test_run(
                test_config,
                dataset_version=pinned_version,
                judge_versions=judge_versions,
                agent_traces=agent_traces,
                name=run_name,
            )
            test_run = prepared.get("test_run") or {}
            test_run_id = test_run.get("id") or ""
            ui_results_url = prepared.get("ui_results_url") or ""

            console.print(f"[dim]Run:[/dim] {test_run_id}")
            judgeval_logger.info(
                f"Created test run {test_run_id} over {len(examples)} examples"
            )

            status = self.wait_for_completion(test_run_id, timeout_seconds, progress)

        items, items_url = self.fetch_items(test_run_id)
        ui_results_url = items_url or ui_results_url

        results = self.build_results(items, agent_traces, pass_condition_fn)

        if pass_condition_fn is not None:
            self.report_success(test_run_id, prepared, items, results)

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
