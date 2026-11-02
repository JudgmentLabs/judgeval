from __future__ import annotations

import httpx
import pytest

from judgeval import Judgeval
from judgeval.exceptions import (
    JudgmentAPIError,
    JudgmentConflictError,
    JudgmentProjectNotFoundError,
    JudgmentValidationError,
)
from judgeval.internal.api.api_client import JudgmentSyncClient, _handle_response


def make_client(monkeypatch: pytest.MonkeyPatch, project_id: str | None) -> Judgeval:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: project_id)
    return Judgeval(
        project_name="demo",
        api_key="api-key",
        organization_id="org-1",
        api_url="https://api.example.com/",
    )


def test_sql_posts_to_project_sql_route(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch, "project-1")
    calls = []
    result = {
        "catalog_version": "v1",
        "columns": [{"name": "n", "type": "UInt64", "nullable": False}],
        "rows": [{"n": 1}],
        "row_count": 1,
        "elapsed_ms": 4,
    }

    def fake_request(self, method, url, payload, params=None):  # type: ignore[no-untyped-def]
        calls.append((method, url, payload, params))
        return result

    monkeypatch.setattr(JudgmentSyncClient, "_request", fake_request)

    assert client.sql("SELECT 1 AS n") == result
    assert calls == [
        (
            "POST",
            "https://api.example.com/v1/projects/project-1/sql",
            {"sql": "SELECT 1 AS n"},
            None,
        )
    ]


def test_sql_requires_a_project_visible_to_the_organization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch, None)

    with pytest.raises(
        JudgmentProjectNotFoundError,
        match="Project 'demo' was not found for this organization",
    ):
        client.sql("SELECT 1")


def test_sql_errors_map_to_typed_sdk_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch, "project-1")

    def raise_status(status_code: int) -> None:
        def fake_request(self, method, url, payload, params=None):  # type: ignore[no-untyped-def]
            raise JudgmentAPIError(status_code, "boom", None, code="SQL_INVALID")

        monkeypatch.setattr(JudgmentSyncClient, "_request", fake_request)

    raise_status(422)
    with pytest.raises(JudgmentValidationError) as invalid:
        client.sql("SELECT 1")
    assert invalid.value.code == "SQL_INVALID"

    raise_status(409)
    with pytest.raises(JudgmentConflictError):
        client.sql("SELECT 1")

    raise_status(500)
    with pytest.raises(JudgmentAPIError) as server_error:
        client.sql("SELECT 1")
    assert type(server_error.value) is JudgmentAPIError


@pytest.mark.parametrize(
    ("retry_after", "expected_seconds"),
    [("2", 2), ("Wed, 21 Oct 2015 07:28:00 GMT", None)],
)
def test_api_error_preserves_code_hint_and_retry_after(
    retry_after: str, expected_seconds: int | None
) -> None:
    response = httpx.Response(
        429,
        json={
            "error": "RATE_LIMITED",
            "message": "Retry later.",
            "hint": "Slow down.",
        },
        headers={"Retry-After": retry_after},
    )

    with pytest.raises(JudgmentAPIError) as raised:
        _handle_response(response)

    assert vars(raised.value) == {
        "status_code": 429,
        "detail": "Retry later.",
        "response": response,
        "code": "RATE_LIMITED",
        "hint": "Slow down.",
        "retry_after_seconds": expected_seconds,
        "_request": None,
    }
