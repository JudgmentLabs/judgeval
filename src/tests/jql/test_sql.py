from __future__ import annotations

import pytest

from judgeval import Judgeval
from judgeval.exceptions import JudgmentAPIError, JudgmentProjectNotFoundError
from judgeval.internal.api.api_client import JudgmentSyncClient

SQL = "SELECT count() AS run_count FROM telemetry.traces"
RESULT = {
    "catalog_version": "1",
    "columns": [{"name": "run_count", "type": "UInt64", "nullable": False}],
    "rows": [{"run_count": 64376}],
    "row_count": 1,
    "elapsed_ms": 2,
}


def client() -> Judgeval:
    return Judgeval(
        project_name="demo",
        api_key="api-key",
        organization_id="org-1",
        api_url="https://api.example.com/",
    )


def test_sql_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: "project-1")
    calls = []

    def request(self, method, url, payload, params=None):
        calls.append((method, url, payload, params))
        return RESULT

    monkeypatch.setattr(JudgmentSyncClient, "_request", request)
    assert {"response": client().sql(SQL), "calls": calls} == {
        "response": RESULT,
        "calls": [
            (
                "POST",
                "https://api.example.com/v1/projects/project-1/sql",
                {"sql": SQL},
                None,
            )
        ],
    }


def test_sql_requires_resolved_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: None)
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(JudgmentSyncClient, "_request", request)
    with pytest.raises(JudgmentProjectNotFoundError) as caught:
        client().sql(SQL)
    assert {"message": str(caught.value), "calls": calls} == {
        "message": "Project 'demo' was not found for this organization; JQL queries require a resolved project.",
        "calls": [],
    }


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 502])
def test_sql_maps_errors_like_query(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    from judgeval.exceptions import map_judgment_api_error

    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: "project-1")
    error = JudgmentAPIError(status, "Rejected", None, code="BAD_SQL")
    expected = map_judgment_api_error(error)

    def request(*args, **kwargs):
        raise error

    monkeypatch.setattr(JudgmentSyncClient, "_request", request)
    with pytest.raises(type(expected)) as caught:
        client().sql(SQL)
    assert (type(caught.value), vars(caught.value), str(caught.value)) == (
        type(expected),
        vars(expected),
        str(expected),
    )


def test_generated_sql_response_contract() -> None:
    from typing import Any, Dict, List
    from judgeval.jql._generated_transport import SqlResponse

    assert SqlResponse.__annotations__ == {
        "catalog_version": str,
        "columns": List[Dict[str, Any]],
        "rows": List[Dict[str, Any]],
        "row_count": int,
        "elapsed_ms": float,
    }
