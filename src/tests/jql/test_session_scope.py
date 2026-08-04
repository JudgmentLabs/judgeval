from __future__ import annotations

import pytest

from judgeval import Judgeval
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.jql import spans


@pytest.mark.parametrize(
    ("scope", "expected_scope"),
    [
        ({"trace_ids": ["trace-1"]}, {"trace_ids": ["trace-1"]}),
        ({"session_ids": ["session-1"]}, {"session_ids": ["session-1"]}),
    ],
)
def test_judgeval_query_sends_scope_outside_jql(
    monkeypatch: pytest.MonkeyPatch,
    scope: dict[str, list[str]],
    expected_scope: dict[str, list[str]],
) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: "project-1")
    calls = []

    def fake_request(self, method, url, payload, params=None):  # type: ignore[no-untyped-def]
        calls.append((method, url, payload, params))
        return {
            "query_id": "q-1",
            "rows": [{"span_id": "span-1"}],
            "row_count": 1,
            "elapsed_ms": 4,
        }

    monkeypatch.setattr(JudgmentSyncClient, "_request", fake_request)
    client = Judgeval(
        project_name="demo",
        api_key="api-key",
        organization_id="org-1",
        api_url="https://api.example.com/",
    )

    response = client.query(spans().rows(), **scope)  # type: ignore[arg-type]

    assert {"calls": calls, "response": response} == {
        "calls": [
            (
                "POST",
                "https://api.example.com/v1/projects/project-1/query",
                {
                    "query": {
                        "op": "query",
                        "source": "spans",
                        "select": {"op": "rows"},
                    },
                    **expected_scope,
                },
                None,
            )
        ],
        "response": {
            "query_id": "q-1",
            "rows": [{"span_id": "span-1"}],
            "row_count": 1,
            "elapsed_ms": 4,
        },
    }


def test_judgeval_query_rejects_trace_and_session_scope_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: "project-1")
    calls = []

    def fake_request(self, method, url, payload, params=None):  # type: ignore[no-untyped-def]
        calls.append((method, url, payload, params))
        raise AssertionError("must not run")

    monkeypatch.setattr(JudgmentSyncClient, "_request", fake_request)
    client = Judgeval(
        project_name="demo",
        api_key="api-key",
        organization_id="org-1",
        api_url="https://api.example.com/",
    )

    with pytest.raises(ValueError) as caught:
        client.query(
            spans().rows(),
            trace_ids=["trace-1"],
            session_ids=["session-1"],
        )

    assert {
        "error": {
            "type": type(caught.value).__name__,
            "message": str(caught.value),
        },
        "calls": calls,
    } == {
        "error": {
            "type": "ValueError",
            "message": "trace_ids and session_ids are mutually exclusive",
        },
        "calls": [],
    }
