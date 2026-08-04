from __future__ import annotations

import pytest

from judgeval import Judgeval
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.jql import spans


def test_judgeval_query_sends_session_scope_outside_jql(
    monkeypatch: pytest.MonkeyPatch,
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

    response = client.query(spans().rows(), session_ids=["session-1"])

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
                    "session_ids": ["session-1"],
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
