from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from judgeval.data.example import Example
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.testing import EvaluatorJudge

EVALUATOR = dedent("""\
    import asyncio

    from judgeval.testing import evaluator, failed, passed, select, Result

    ev = evaluator()

    def final_output(trace):
        return select(trace, "SELECT span_id, output FROM telemetry.spans WHERE span_name = 'output'")[0]

    @ev.check("has_output")
    def has_output(case, trace):
        return passed() if final_output(trace)["output"] else failed("empty reply")

    @ev.check("mentions_input")
    def mentions_input(case, trace):
        return case.input["topic"] in final_output(trace)["output"]

    @ev.check("tone", labels=["calm", "curt"])
    def tone(case, trace):
        return Result("calm", reason="no exclamation marks", evidence=[{"span_id": "s1", "quote": "Sure"}])

    @ev.check("tool_budget", applies="when_declared")
    def tool_budget(case, trace, settings):
        return len(select(trace, "SELECT span_id FROM telemetry.spans WHERE span_kind = 'tool'")) <= settings["max"]

    @ev.check("crashes", gates=False)
    def crashes(case, trace):
        return {}["missing"]

    @ev.check("wrong_kind")
    def wrong_kind(case, trace):
        return "yes"

    @ev.check("slow", gates=False)
    async def slow(case, trace):
        await asyncio.sleep(5)
        return True

    ev.metric("tool_calls", lambda case, trace: 2, rollup="mean", better="lower", budget=3)
""")

ROWS = {
    "SELECT span_id, output FROM telemetry.spans WHERE span_name = 'output'": [
        {"span_id": "s1", "output": "Sure, billing questions go to the billing team."}
    ],
    "SELECT span_id FROM telemetry.spans WHERE span_kind = 'tool'": [
        {"span_id": "t1"},
        {"span_id": "t2"},
    ],
}


def ok(name: str, value: Any, passed: Any, **extra: Any) -> dict[str, Any]:
    return {
        "check_name": name,
        "value": value,
        "passed": passed,
        "probability": None,
        "reason": None,
        "evidence": [],
        "cost_usd": None,
        "error": None,
        **extra,
    }


def error(name: str, message: str) -> dict[str, Any]:
    return {**ok(name, None, False), "error": message}


@pytest.fixture
def entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type[EvaluatorJudge]:
    path = tmp_path / "evaluator.py"
    path.write_text(EVALUATOR)
    spec = importlib.util.spec_from_file_location("evaluator_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("JUDGMENT_TESTING_API_URL", "https://api.example")
    monkeypatch.setenv("JUDGMENT_TESTING_TOKEN", "run-token")

    class Entry(EvaluatorJudge):
        check_timeout_s = 0.2

    Entry.module = module
    return Entry


def test_score_runs_requested_checks(
    entry: type[EvaluatorJudge], monkeypatch: pytest.MonkeyPatch
):
    requests: list[tuple[str, str, Any, str, str]] = []

    def fake_request(
        self: JudgmentSyncClient, method: str, url: str, payload: Any
    ) -> Any:
        requests.append((method, url, payload, self.api_key, self.organization_id))
        return {"rows": ROWS[payload["sql"]]}

    monkeypatch.setattr(JudgmentSyncClient, "_request", fake_request)
    example = Example.create(
        case_id="c1",
        trace_id="t-abc",
        organization_id="org-1",
        project_id="proj-1",
        input={"topic": "billing"},
        metadata={"checks": {"tool_budget": {"max": 1}}},
        checks=[
            "has_output",
            "mentions_input",
            "tone",
            "tool_budget",
            "crashes",
            "wrong_kind",
            "slow",
            "tool_calls",
            "undeclared",
        ],
    )

    response = asyncio.run(entry().score(example))

    assert response.results == [
        ok("has_output", True, True),
        ok("mentions_input", True, True),
        ok(
            "tone",
            "calm",
            None,
            reason="no exclamation marks",
            evidence=[{"span_id": "s1", "quote": "Sure"}],
        ),
        ok("tool_budget", False, False),
        error("crashes", "KeyError: 'missing'"),
        error("wrong_kind", "pass_fail check returned str, not bool"),
        error("slow", "timed out after 0.2s"),
        ok("tool_calls", 2.0, True),
        error("undeclared", "not declared in the evaluator"),
    ]
    # Gating: has_output, mentions_input, tool_budget, wrong_kind, tool_calls.
    assert (response.value, response.reason) == (False, "3/5 gating checks passed")
    assert {r[:2] for r in requests} == {
        ("POST", "https://api.example/v1/projects/proj-1/sql")
    }
    assert {(r[2]["trace_ids"][0], r[3], r[4]) for r in requests} == {
        ("t-abc", "run-token", "org-1")
    }
