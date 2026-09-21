from __future__ import annotations

import httpx
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


def test_discover_schema_returns_reference_without_resolved_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: None)
    reference = "# Judgment SQL\n\n## telemetry.traces\ntrace_id: String — Trace ID\n"
    calls = []

    def request(self, method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(200, json={"schema": reference})

    monkeypatch.setattr(httpx.Client, "request", request)
    assert {"reference": client().discover_schema(), "calls": calls} == {
        "reference": reference,
        "calls": [
            (
                "GET",
                "https://api.example.com/v1/sql/schema",
                {
                    "params": {},
                    "headers": {
                        "Authorization": "Bearer api-key",
                        "X-Organization-Id": "org-1",
                        "Content-Type": "application/json",
                    },
                },
            )
        ],
    }


@pytest.mark.parametrize(
    "sql_response",
    [
        RESULT,
        {
            "catalog_version": "1",
            "columns": [
                {"name": "run_count", "type": "UInt64", "nullable": False},
                {"name": "values", "type": "Array(Int64)", "nullable": False},
            ],
            "rows": [
                {
                    "run_count": "9007199254740993",
                    "values": ["-9223372036854775808", 42],
                }
            ],
            "row_count": 1,
            "elapsed_ms": 2,
        },
    ],
)
def test_sql_forwarding(monkeypatch: pytest.MonkeyPatch, sql_response: dict) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: "project-1")
    calls = []

    def request(self, method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(200, json=sql_response)

    monkeypatch.setattr(httpx.Client, "request", request)
    assert {"response": client().sql(SQL), "calls": calls} == {
        "response": sql_response,
        "calls": [
            (
                "POST",
                "https://api.example.com/v1/projects/project-1/sql",
                {
                    "json": {"sql": SQL},
                    "params": None,
                    "headers": {
                        "Authorization": "Bearer api-key",
                        "X-Organization-Id": "org-1",
                        "Content-Type": "application/json",
                    },
                },
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
        "message": "Project 'demo' was not found for this organization; Public queries require a resolved project.",
        "calls": [],
    }


@pytest.mark.parametrize("status", [422, 429])
def test_sql_maps_errors_like_query(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    from judgeval.exceptions import map_judgment_api_error

    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: "project-1")
    error = JudgmentAPIError(
        status,
        "Rejected",
        None,
        code="BAD_SQL",
        hint="Check the query.",
        retry_after_seconds=2 if status == 429 else None,
    )
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
