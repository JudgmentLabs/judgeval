from __future__ import annotations

import inspect

import httpx
import pytest

import judgeval.jql as jql
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
    module_builder_ops = {
        name
        for name, value in vars(jql).items()
        if inspect.isfunction(value)
        and value.__module__ == jql.__name__
        and not name.startswith("_")
    } - {"all_", "any_", "not_", "traces", "spans", "sessions", "to_json"}
    # Normalize Python-safe aliases and source constructors to canonical JQL ops.
    module_builder_ops.update({"all", "any", "not", "query"})

    fluent_builder_ops = {
        name
        for builder in (jql.QueryBuilder, jql.PipelineBuilder)
        for name, value in vars(builder).items()
        if callable(value) and not name.startswith("_")
    } - {"last", "since", "between", "pipe", "to_json"}

    assert module_builder_ops | fluent_builder_ops == set(SUPPORTED_OPS)


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

    assert presentation == {
        "title": "Spend by model",
        "columns": [
            {"key": "model"},
            {"key": "total", "format": "currency_usd"},
        ],
        "op": "table",
        "query": {
            "op": "query",
            "source": "spans",
            "time": {"last": "7d"},
            "pipe": [
                {
                    "op": "derive",
                    "cols": {"expensive": {"op": "col", "name": "cost"}},
                },
                {
                    "op": "where",
                    "filter": {
                        "op": "gte",
                        "field": "expensive",
                        "value": 1,
                    },
                },
                {
                    "op": "summarize",
                    "by": "model",
                    "aggs": {
                        "total": {
                            "op": "agg_expr",
                            "func": "sum",
                            "field": "cost",
                        }
                    },
                },
                {"op": "take", "n": 10},
            ],
        },
    }


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
    assert response == {
        "query_id": "q-1",
        "rows": [{"trace_id": "trace-1"}],
        "row_count": 1,
        "elapsed_ms": 4,
    }


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

    assert vars(raised.value) == {
        "status_code": 429,
        "detail": "Retry later.",
        "response": response,
        "code": "JQL_RATE_LIMITED",
        "hint": "Slow down.",
        "retry_after_seconds": 2,
        "_request": None,
    }


def test_api_error_ignores_http_date_retry_after() -> None:
    response = httpx.Response(
        429,
        json={
            "error": "JQL_RATE_LIMITED",
            "message": "Retry later.",
            "hint": "Slow down.",
        },
        headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"},
    )

    with pytest.raises(JudgmentAPIError) as raised:
        _handle_response(response)

    assert vars(raised.value) == {
        "status_code": 429,
        "detail": "Retry later.",
        "response": response,
        "code": "JQL_RATE_LIMITED",
        "hint": "Slow down.",
        "retry_after_seconds": None,
        "_request": None,
    }
