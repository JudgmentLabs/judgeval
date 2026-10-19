"""Test suites: versioned test cases scored by judges against your agent's traces.

A test suite holds cases. Each case has an ``input`` for your agent and an
``expected`` mapping that names the judges to run on the case's trace and
what each judge's output must compare to. Edit cases in the suite's draft,
publish a version, then start runs against it.
"""

from __future__ import annotations

import re
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
    test_suite_version: int
    cases: List[StartedCase]


class RunSummary(TypedDict):
    run_id: str
    test_suite_id: str
    test_suite_version: int
    scored_from_run_id: Optional[str]
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


def _segment(value: str) -> str:
    # Ids land in the URL path, so "..", "?" or "#" could reach another route.
    if not re.fullmatch(r"[\w-]+", value):
        raise ValueError(f"Invalid id in API path: {value!r}")
    return value


class TestSuitesClient:
    """HTTP calls for the test suite routes, named as the generated client names them.

    TODO: delete once ``JudgmentSyncClient`` is generated from the committed /v1
    spec (judgeval#805); the factories then call its methods directly.
    """

    __slots__ = ("_client",)
    __test__ = False  # not a pytest test class

    def __init__(self, client: JudgmentSyncClient):
        self._client = client

    def _call(
        self,
        method: Method,
        path: str,
        payload: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = url_for(f"/v1/projects/{path}", self._client.base_url)
        try:
            response = self._client._request(
                method,  # type: ignore[arg-type]
                url,
                {} if payload is None else payload,
                params,
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(e) from e
        return cast(Dict[str, Any], response)

    def post_projects_test_suites(
        self, project_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._call("POST", f"{_segment(project_id)}/test-suites", payload)

    def get_projects_test_suites(
        self, project_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._call("GET", f"{_segment(project_id)}/test-suites", params=params)

    def delete_projects_test_suites_by_test_suite_id(
        self, project_id: str, test_suite_id: str
    ) -> Dict[str, Any]:
        return self._call(
            "DELETE", f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}"
        )

    def get_projects_test_suites_by_test_suite_id_cases(
        self, project_id: str, test_suite_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._call(
            "GET",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}/cases",
            params=params,
        )

    def post_projects_test_suites_by_test_suite_id_cases(
        self, project_id: str, test_suite_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._call(
            "POST",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}/cases",
            payload,
        )

    def put_projects_test_suites_by_test_suite_id_cases_by_case_id(
        self,
        project_id: str,
        test_suite_id: str,
        case_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._call(
            "PUT",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}"
            f"/cases/{_segment(case_id)}",
            payload,
        )

    def delete_projects_test_suites_by_test_suite_id_cases_by_case_id(
        self, project_id: str, test_suite_id: str, case_id: str
    ) -> Dict[str, Any]:
        return self._call(
            "DELETE",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}"
            f"/cases/{_segment(case_id)}",
        )

    def post_projects_test_suites_by_test_suite_id_draft(
        self, project_id: str, test_suite_id: str
    ) -> Dict[str, Any]:
        return self._call(
            "POST",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}/draft",
        )

    def post_projects_test_suites_by_test_suite_id_publish(
        self, project_id: str, test_suite_id: str
    ) -> Dict[str, Any]:
        return self._call(
            "POST",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}/publish",
        )

    def post_projects_test_suites_by_test_suite_id_discard(
        self, project_id: str, test_suite_id: str
    ) -> Dict[str, Any]:
        return self._call(
            "POST",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}/discard",
        )

    def post_projects_test_suites_by_test_suite_id_runs(
        self, project_id: str, test_suite_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._call(
            "POST",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}/runs",
            payload,
        )

    def get_projects_test_suites_by_test_suite_id_runs(
        self, project_id: str, test_suite_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._call(
            "GET",
            f"{_segment(project_id)}/test-suites/{_segment(test_suite_id)}/runs",
            params=params,
        )

    def get_projects_test_suites_runs_by_run_id(
        self, project_id: str, run_id: str
    ) -> Dict[str, Any]:
        return self._call(
            "GET", f"{_segment(project_id)}/test-suites/runs/{_segment(run_id)}"
        )

    def get_projects_test_suites_runs_by_run_id_results(
        self, project_id: str, run_id: str
    ) -> Dict[str, Any]:
        return self._call(
            "GET",
            f"{_segment(project_id)}/test-suites/runs/{_segment(run_id)}/results",
        )

    def post_projects_test_suites_runs_by_run_id_score(
        self, project_id: str, run_id: str
    ) -> Dict[str, Any]:
        return self._call(
            "POST",
            f"{_segment(project_id)}/test-suites/runs/{_segment(run_id)}/score",
        )


class ProjectScopedFactory:
    """Base for the test suite factories: the project they act on and its client."""

    __slots__ = ("_api", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._api = TestSuitesClient(client)
        self._project_id = project_id
        self._project_name = project_name

    @property
    def _project(self) -> str:
        if not self._project_id:
            raise JudgmentProjectNotFoundError(
                f"Project '{self._project_name}' not found"
            )
        return self._project_id


def _without_none(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


class TestSuitesFactory(ProjectScopedFactory):
    """Create test suites and edit their cases. Access via ``client.test_suites``.

    Cases change only in a suite's draft (version + 1). New cases go straight
    into the draft; to edit or remove published cases, call ``create_draft``
    first.
    ``publish`` freezes the draft as the next version.

    Examples:
        ```python
        suite = client.test_suites.create(name="support-agent-regressions")
        client.test_suites.add_cases(
            suite["test_suite_id"],
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
        client.test_suites.publish(suite["test_suite_id"])
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
        """Create an empty test suite."""
        return self._api.post_projects_test_suites(
            self._project,
            _without_none({"name": name, "description": description, "labels": labels}),
        )

    def list(self, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        """List the project's test suites, newest first (``test_suites`` and ``next_offset``)."""
        return self._api.get_projects_test_suites(
            self._project, {"offset": offset, "limit": limit}
        )

    def delete(self, test_suite_id: str) -> Dict[str, Any]:
        """Delete a test suite with all its versions, runs and results."""
        return self._api.delete_projects_test_suites_by_test_suite_id(
            self._project, test_suite_id
        )

    def list_cases(
        self,
        test_suite_id: str,
        version: Optional[int | Literal["draft"]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List the cases in a published version (default: latest) or the draft."""
        params: Dict[str, Any] = {"offset": offset, "limit": limit}
        if version is not None:
            params["version"] = str(version)
        return self._api.get_projects_test_suites_by_test_suite_id_cases(
            self._project, test_suite_id, params
        )

    def add_cases(
        self, test_suite_id: str, cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add cases to the draft. Returns their ``case_ids`` in order.

        Each case is ``{"input": ..., "expected": {judge_id: check}, "tags": [...],
        "name": ..., "metadata": {...}}``; only ``input`` is required. A check is
        ``{"judge_major_version", "judge_minor_version", "op", "expected"}``,
        where ``op`` is one of eq, ne, lt, lte, gt, gte, in, not_in.
        """
        return self._api.post_projects_test_suites_by_test_suite_id_cases(
            self._project, test_suite_id, {"cases": cases}
        )

    def update_case(
        self, test_suite_id: str, case_id: str, case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Replace a case in the draft. Call ``create_draft`` first for published cases."""
        return self._api.put_projects_test_suites_by_test_suite_id_cases_by_case_id(
            self._project, test_suite_id, case_id, case
        )

    def remove_case(self, test_suite_id: str, case_id: str) -> Dict[str, Any]:
        """Remove a case from the draft. Call ``create_draft`` first for published cases."""
        return self._api.delete_projects_test_suites_by_test_suite_id_cases_by_case_id(
            self._project, test_suite_id, case_id
        )

    def create_draft(self, test_suite_id: str) -> Dict[str, Any]:
        """Copy the published cases into the draft so they can be edited."""
        return self._api.post_projects_test_suites_by_test_suite_id_draft(
            self._project, test_suite_id
        )

    def publish(self, test_suite_id: str) -> Dict[str, Any]:
        """Freeze the draft as the next version. Returns the new ``version``."""
        return self._api.post_projects_test_suites_by_test_suite_id_publish(
            self._project, test_suite_id
        )

    def discard(self, test_suite_id: str) -> Dict[str, Any]:
        """Throw away the draft's changes."""
        return self._api.post_projects_test_suites_by_test_suite_id_discard(
            self._project, test_suite_id
        )


class TestRunsFactory(ProjectScopedFactory):
    """Run test suites and read results. Access via ``client.test_runs``.

    A run returns the cases to execute. Run your agent once per case on your
    normal tracer with the run and case ids on the trace's root span, using
    ``Tracer.test_case`` or by sending ``case["headers"]`` to a deployed agent.
    Each case is scored as soon as its trace arrives.

    Examples:
        ```python
        run = client.test_runs.start(
            test_suite_id,
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
        test_suite_id: str,
        case_filter: Optional[Dict[str, Any]] = None,
        candidate: Optional[str] = None,
        candidate_meta: Optional[Dict[str, Any]] = None,
        trigger: Optional[Literal["manual", "ci", "schedule", "api"]] = None,
    ) -> StartedRun:
        """Start a run of the test suite's latest published version.

        Returns ``run_id``, ``test_suite_version`` and ``cases`` (each with
        ``case_id``, ``input`` and tagging ``headers``).
        """
        response = self._api.post_projects_test_suites_by_test_suite_id_runs(
            self._project,
            test_suite_id,
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

    def list(
        self, test_suite_id: str, offset: int = 0, limit: int = 50
    ) -> Dict[str, Any]:
        """List a test suite's runs, newest first (``runs`` and ``next_offset``)."""
        return self._api.get_projects_test_suites_by_test_suite_id_runs(
            self._project, test_suite_id, {"offset": offset, "limit": limit}
        )

    def get(self, run_id: str) -> RunSummary:
        """Get a run and how many of its cases have a trace, are scored and passed."""
        return cast(
            RunSummary,
            self._api.get_projects_test_suites_runs_by_run_id(self._project, run_id),
        )

    def results(self, run_id: str) -> RunResults:
        """Get each case's trace, outcome and per-judge check results."""
        return cast(
            RunResults,
            self._api.get_projects_test_suites_runs_by_run_id_results(
                self._project, run_id
            ),
        )

    def score(self, run_id: str) -> Dict[str, Any]:
        """Trigger scoring of an existing run's traces with the latest published version.

        Returns the new run's ``run_id`` right away; its cases are scored in the
        background, so poll ``get`` for progress. The agent doesn't run again.
        """
        return self._api.post_projects_test_suites_runs_by_run_id_score(
            self._project, run_id
        )


__all__ = (
    "CaseResult",
    "CheckResult",
    "RunResults",
    "RunSummary",
    "StartedCase",
    "StartedRun",
    "ProjectScopedFactory",
    "TestRunsFactory",
    "TestSuitesClient",
    "TestSuitesFactory",
)
