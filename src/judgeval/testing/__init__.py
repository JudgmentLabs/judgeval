"""Tests: versioned test cases scored by judges against your agent's traces.

A test holds cases. Each case has an ``input`` for your agent and an
``expected`` mapping that names the judges to run on the case's trace and
what each judge's output must compare to. Edit cases in the test's draft,
publish a version, then start runs against it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

from judgeval.exceptions import (
    JudgmentAPIError,
    JudgmentProjectNotFoundError,
    map_judgment_api_error,
)
from judgeval.internal.api import JudgmentSyncClient
from judgeval.utils.url import url_for

Method = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


class StartedCase(TypedDict):
    case_id: str
    input: Any
    #: Headers that tag a deployed agent's trace with this run and case.
    headers: Dict[str, str]


class StartedRun(TypedDict):
    run_id: str
    test_version: int
    cases: List[StartedCase]


class RunSummary(TypedDict):
    run_id: str
    test_id: str
    test_version: int
    rescored_from_run_id: Optional[str]
    candidate: Optional[str]
    candidate_meta: Dict[str, Any]
    trigger: str
    case_filter: Dict[str, Any]
    cases_total: int
    cases_with_output: int
    cases_scored: int
    cases_passed: int
    created_at: str


class CheckResult(TypedDict):
    judge_id: str
    judge_major_version: int
    judge_minor_version: int
    op: str
    expected: Any
    judge_output: Any
    passed: bool
    reason: Optional[str]
    error_message: Optional[str]


class CaseResult(TypedDict):
    case_id: str
    trace_id: Optional[str]
    scored_at: Optional[str]
    passed: Optional[bool]
    checks: List[CheckResult]


class RunResults(TypedDict):
    results: List[CaseResult]


class _ProjectScoped:
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

    def _call(
        self,
        method: Method,
        path: str,
        payload: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._project_id:
            raise JudgmentProjectNotFoundError(
                f"Project '{self._project_name}' not found"
            )
        url = url_for(
            f"/v1/projects/{self._project_id}/tests{path}", self._client.base_url
        )
        try:
            # TODO: switch to the generated client once it covers these routes (judgeval#805).
            response = self._client._request(
                method,  # type: ignore[arg-type]
                url,
                {} if payload is None else payload,
                params,
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(e) from e
        return cast(Dict[str, Any], response)


def _without_none(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


class TestsFactory(_ProjectScoped):
    """Create tests and edit their cases. Access via ``client.tests``.

    Cases change only in a test's draft (version + 1). New cases go straight
    into the draft; to edit or remove published cases, call ``create_draft``
    first.
    ``publish`` freezes the draft as the next version.

    Examples:
        ```python
        test = client.tests.create(name="support-agent-regressions")
        client.tests.add_cases(
            test["test_id"],
            [
                {
                    "input": {"prompt": "I want a refund"},
                    "expected": {
                        "<judge-id>": {
                            "judge_major_version": 1,
                            "judge_minor_version": 0,
                            "op": "eq",
                            "expected": "refund",
                        }
                    },
                    "tags": ["billing"],
                }
            ],
        )
        client.tests.publish(test["test_id"])
        ```
    """

    __slots__ = ()
    __test__ = False  # not a pytest test class

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create an empty test."""
        return self._call(
            "POST",
            "",
            _without_none({"name": name, "description": description, "labels": labels}),
        )

    def list(self, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        """List the project's tests, newest first (``tests`` and ``next_offset``)."""
        return self._call("GET", "", params={"offset": offset, "limit": limit})

    def delete(self, test_id: str) -> Dict[str, Any]:
        """Delete a test with all its versions, runs and results."""
        return self._call("DELETE", f"/{test_id}")

    def list_cases(
        self,
        test_id: str,
        version: Optional[int | Literal["draft"]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List the cases in a published version (default: latest) or the draft."""
        params: Dict[str, Any] = {"offset": offset, "limit": limit}
        if version is not None:
            params["version"] = str(version)
        return self._call("GET", f"/{test_id}/cases", params=params)

    def add_cases(self, test_id: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add cases to the draft. Returns their ``case_ids`` in order.

        Each case is ``{"input": ..., "expected": {judge_id: check}, "tags": [...],
        "name": ..., "metadata": {...}}``; only ``input`` is required. A check is
        ``{"judge_major_version", "judge_minor_version", "op", "expected"}``,
        where ``op`` is one of eq, ne, lt, lte, gt, gte, in, not_in.
        """
        return self._call("POST", f"/{test_id}/cases", {"cases": cases})

    def update_case(
        self, test_id: str, case_id: str, case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Replace a case in the draft. Call ``create_draft`` first for published cases."""
        return self._call("PUT", f"/{test_id}/cases/{case_id}", case)

    def remove_case(self, test_id: str, case_id: str) -> Dict[str, Any]:
        """Remove a case from the draft. Call ``create_draft`` first for published cases."""
        return self._call("DELETE", f"/{test_id}/cases/{case_id}")

    def create_draft(self, test_id: str) -> Dict[str, Any]:
        """Copy the published cases into the draft so they can be edited."""
        return self._call("POST", f"/{test_id}/draft")

    def publish(self, test_id: str) -> Dict[str, Any]:
        """Freeze the draft as the next version. Returns the new ``version``."""
        return self._call("POST", f"/{test_id}/publish")

    def discard(self, test_id: str) -> Dict[str, Any]:
        """Throw away the draft's changes."""
        return self._call("POST", f"/{test_id}/discard")


class TestRunsFactory(_ProjectScoped):
    """Run tests and read results. Access via ``client.test_runs``.

    A run returns the cases to execute. Run your agent once per case on your
    normal tracer with the run and case ids on the trace's root span, using
    ``Tracer.test_case`` or by sending ``case["headers"]`` to a deployed agent.
    Each case is scored as soon as its trace arrives.

    Examples:
        ```python
        run = client.test_runs.start(
            test_id,
            case_filter={"kind": "tags", "tags": ["billing"]},
            candidate=os.environ["GIT_SHA"],
        )
        for case in run["cases"]:
            with Tracer.test_case(run["run_id"], case["case_id"]):
                support_agent(case["input"])

        results = client.test_runs.results(run["run_id"])
        ```
    """

    __slots__ = ()
    __test__ = False  # not a pytest test class

    def start(
        self,
        test_id: str,
        case_filter: Optional[Dict[str, Any]] = None,
        candidate: Optional[str] = None,
        candidate_meta: Optional[Dict[str, Any]] = None,
        trigger: Optional[Literal["manual", "ci", "schedule", "api"]] = None,
    ) -> StartedRun:
        """Start a run of the test's latest published version.

        Returns ``run_id``, ``test_version`` and ``cases`` (each with
        ``case_id``, ``input`` and tagging ``headers``).
        """
        response = self._call(
            "POST",
            f"/{test_id}/runs",
            _without_none(
                {
                    "case_filter": case_filter,
                    "candidate": candidate,
                    "candidate_meta": candidate_meta,
                    "trigger": trigger,
                }
            ),
        )
        return cast(StartedRun, response)

    def list(self, test_id: str, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        """List a test's runs, newest first (``runs`` and ``next_offset``)."""
        return self._call(
            "GET", f"/{test_id}/runs", params={"offset": offset, "limit": limit}
        )

    def get(self, run_id: str) -> RunSummary:
        """Get a run and how many of its cases have a trace, are scored and passed."""
        return cast(RunSummary, self._call("GET", f"/runs/{run_id}"))

    def results(self, run_id: str) -> RunResults:
        """Get each case's trace, outcome and per-judge check results."""
        return cast(RunResults, self._call("GET", f"/runs/{run_id}/results"))

    def rescore(self, run_id: str) -> Dict[str, Any]:
        """Score a run's traces again with the latest published version.

        Returns the new run's ``run_id`` right away; its cases are scored in the
        background, so poll ``get`` for progress. The agent doesn't run again.
        """
        return self._call("POST", f"/runs/{run_id}/rescore")


__all__ = (
    "CaseResult",
    "CheckResult",
    "RunResults",
    "RunSummary",
    "StartedCase",
    "StartedRun",
    "TestRunsFactory",
    "TestsFactory",
)
