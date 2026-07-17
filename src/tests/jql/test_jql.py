from __future__ import annotations

import httpx
import pytest

from judgeval import Judgeval
from judgeval.exceptions import JudgmentAPIError
from judgeval.internal.api.api_client import JudgmentSyncClient, _handle_response
from judgeval.jql import (
    agg_expr,
    col,
    eq,
    gte,
    spans,
    traces,
)
from judgeval.jql._generated_contract import SUPPORTED_OPS


def test_generated_contract_covers_the_builder_surface() -> None:
    assert {
        "agg",
        "agg_expr",
        "all",
        "ancestor_of",
        "any",
        "any_span",
        "any_trace",
        "arith",
        "at_least",
        "attr",
        "bucket",
        "chart",
        "cited_by",
        "col",
        "cost",
        "count",
        "derive",
        "descendant_of",
        "discovery",
        "duration",
        "eq",
        "every_span",
        "every_trace",
        "grep",
        "gt",
        "gte",
        "ids",
        "judge",
        "judged",
        "lt",
        "lte",
        "model",
        "name",
        "ne",
        "no_span",
        "no_trace",
        "not",
        "over_scores",
        "over_spans",
        "over_traces",
        "pick",
        "query",
        "ranked",
        "recent",
        "rg",
        "rows",
        "sort",
        "status",
        "summarize",
        "table",
        "take",
        "tokens",
        "top",
        "trend",
        "where",
    } == set(SUPPORTED_OPS)


def test_session_to_trace_ids_uses_canonical_json() -> None:
    assert traces().where(eq("session", "session-1")).ids().to_json() == {
        "op": "query",
        "source": "traces",
        "filter": {"op": "eq", "field": "session", "value": "session-1"},
        "select": {"op": "ids"},
    }


def test_pipeline_and_presentation_are_project_free() -> None:
    query = (
        spans()
        .last("7d")
        .pipe()
        .derive({"expensive": col("cost")})
        .where(gte("expensive", 1))
        .summarize({"total": agg_expr("sum", "cost")}, by="model")
        .take(10)
    )
    presentation = query.table(
        title="Spend by model",
        columns=[{"key": "model"}, {"key": "total", "format": "currency_usd"}],
    )

    assert presentation["op"] == "table"
    assert presentation["query"]["pipe"][-1] == {"op": "take", "n": 10}
    assert "organization_id" not in presentation["query"]
    assert "project_id" not in presentation["query"]


def test_judgeval_query_sends_only_public_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: "project-1")
    calls = []

    def fake_request(self, method, url, payload, params=None):  # type: ignore[no-untyped-def]
        calls.append((method, url, payload, params))
        return {
            "query_id": "q-1",
            "rows": [{"trace_id": "trace-1"}],
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

    response = client.query(traces().where(eq("session", "session-1")).ids(), limit=25)

    assert calls == [
        (
            "POST",
            "https://api.example.com/v1/projects/project-1/query",
            {
                "query": {
                    "op": "query",
                    "source": "traces",
                    "filter": {
                        "op": "eq",
                        "field": "session",
                        "value": "session-1",
                    },
                    "select": {"op": "ids"},
                },
                "limit": 25,
            },
            None,
        )
    ]
    assert response["rows"] == [{"trace_id": "trace-1"}]


def test_api_error_preserves_code_hint_and_retry_after() -> None:
    response = httpx.Response(
        429,
        json={
            "error": "JQL_RATE_LIMITED",
            "message": "Retry later.",
            "hint": "Slow down.",
        },
        headers={"Retry-After": "2"},
    )

    with pytest.raises(JudgmentAPIError) as raised:
        _handle_response(response)

    assert raised.value.code == "JQL_RATE_LIMITED"
    assert raised.value.hint == "Slow down."
    assert raised.value.retry_after_seconds == 2
