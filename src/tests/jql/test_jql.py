from __future__ import annotations

import inspect

import httpx
import pytest

import judgeval.jql as jql
from judgeval import Judgeval
from judgeval.exceptions import JudgmentAPIError, JudgmentProjectNotFoundError
from judgeval.internal.api.api_client import JudgmentSyncClient, _handle_response
from judgeval.jql import (
    agg_expr,
    at_least,
    col,
    discovery,
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


def test_builders_reject_ops_missing_from_the_generated_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jql, "SUPPORTED_OPS", ())

    with pytest.raises(ValueError, match="Unsupported JQL operation: eq"):
        eq("status", "success")


def test_builder_discriminators_cannot_be_overridden() -> None:
    assert jql.cost(op=1.0)["op"] == "cost"
    assert discovery("judges", op="evil")["op"] == "discovery"
    with pytest.raises(TypeError):
        traces().ranked(op="evil")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        traces().pipe().pick(op="evil")  # type: ignore[call-arg]


def test_ranked_and_pick_emit_contract_fields() -> None:
    assert traces().ranked(by="cost", pick=(2, 5), within="session").to_json()[
        "select"
    ] == {"op": "ranked", "by": "cost", "pick": [2, 5], "within": "session"}
    assert spans().pipe().pick(by="cost", n=3, per="model", reverse=True).to_json()[
        "pipe"
    ] == [{"op": "pick", "by": "cost", "n": 3, "per": "model", "reverse": True}]
    with pytest.raises(TypeError):
        traces().ranked(bye="cost")  # type: ignore[call-arg]


def test_empty_pipeline_is_omitted_and_select_cannot_be_discarded() -> None:
    assert spans().pipe().to_json() == {"op": "query", "source": "spans"}

    with pytest.raises(ValueError, match=r"pipe\(\) cannot follow a select"):
        traces().ids().pipe()


def test_empty_filters_are_preserved_consistently() -> None:
    assert traces({}).to_json()["filter"] == {}
    assert at_least(1, "spans", where={})["where"] == {}
    assert agg_expr("count", where={})["where"] == {}


def test_star_exports_do_not_shadow_python_builtins() -> None:
    assert "all" not in jql.__all__
    assert "any" not in jql.__all__
    assert jql.all is jql.all_
    assert jql.any is jql.any_


def test_session_to_trace_ids_uses_canonical_json() -> None:
    assert traces().where(eq("session", "session-1")).ids().to_json() == {
        "op": "query",
        "source": "traces",
        "filter": {"op": "eq", "field": "session", "value": "session-1"},
        "select": {"op": "ids"},
    }


def test_pipeline_and_presentation_use_canonical_json() -> None:
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


def test_present_normalizes_builder_input_and_discover_sends_limit_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: "project-1")
    calls = []

    def fake_request(self, method, url, payload, params=None):  # type: ignore[no-untyped-def]
        calls.append((url, payload))
        return {}

    monkeypatch.setattr(JudgmentSyncClient, "_request", fake_request)
    client = Judgeval(
        project_name="demo",
        api_key="api-key",
        organization_id="org-1",
        api_url="https://api.example.com/",
    )

    client.present(traces().ids(), limit=10)
    client.discover("judges", limit=25)

    assert calls == [
        (
            "https://api.example.com/v1/projects/project-1/query/presentation",
            {
                "query": {
                    "op": "query",
                    "source": "traces",
                    "select": {"op": "ids"},
                },
                "limit": 10,
            },
        ),
        (
            "https://api.example.com/v1/projects/project-1/query",
            {
                "query": {"op": "discovery", "kind": "judges"},
                "limit": 25,
            },
        ),
    ]


def test_jql_requires_a_project_visible_to_the_organization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("judgeval.judgeval.resolve_project_id", lambda *_: None)
    client = Judgeval(
        project_name="missing",
        api_key="api-key",
        organization_id="org-1",
        api_url="https://api.example.com/",
    )

    with pytest.raises(
        JudgmentProjectNotFoundError,
        match="Project 'missing' was not found for this organization",
    ):
        client.query(traces())


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
