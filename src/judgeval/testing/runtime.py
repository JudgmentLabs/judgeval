"""Run an evaluator's checks on one test case's output trace.

The Modal code-judge worker loads ``Entry``, a generated subclass of
``EvaluatorJudge`` bound to the evaluator module, and awaits ``score(example)``.
The example carries the case, the trace id and the names of the checks that
apply to the case; the response's ``results`` hold one entry per check.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, ClassVar, Optional, Union

from judgeval.data.example import Example
from judgeval.hosted.responses import BaseResponse
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.judges.judge import Judge
from judgeval.testing.checks import Check, parse_checks
from judgeval.utils.url import url_for

CheckFn = Callable[..., Any]
Value = Union[bool, int, float, str]


class Evaluator:
    """Registry of an evaluator file's checks. Options live in the source; the parser reads them."""

    def __init__(self) -> None:
        self.checks: dict[str, CheckFn] = {}

    def check(self, name: str, **_options: Any) -> Callable[[CheckFn], CheckFn]:
        def register(fn: CheckFn) -> CheckFn:
            self.checks[name] = fn
            return fn

        return register

    def metric(self, name: str, fn: CheckFn, **_options: Any) -> CheckFn:
        self.checks[name] = fn
        return fn


def evaluator() -> Evaluator:
    return Evaluator()


@dataclass(frozen=True)
class Case:
    case_id: str
    input: Any
    name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trace:
    """The case's output trace. Spans are read with `select`, never shipped to the sandbox."""

    id: str
    project_id: str
    client: JudgmentSyncClient


@dataclass(frozen=True)
class Result:
    value: Value
    passed: Optional[bool] = None
    reason: Optional[str] = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    probability: Optional[float] = None


def passed(reason: Optional[str] = None) -> Result:
    return Result(True, True, reason)


def failed(reason: Optional[str] = None) -> Result:
    return Result(False, False, reason)


def select(trace: Trace, sql: str) -> list[dict[str, Any]]:
    """Run one read-only virtual SQL SELECT over this trace's spans; returns the rows."""
    response = trace.client._request(
        "POST",
        url_for(f"/v1/projects/{trace.project_id}/sql", trace.client.base_url),
        {"sql": sql, "trace_ids": [trace.id]},
    )
    return list(response["rows"])


class EvaluatorResponse(BaseResponse):
    """`value` is whether every gating check that ran passed; `results` has one entry per check."""

    value: bool
    results: list[dict[str, Any]]


class EvaluatorJudge(Judge[EvaluatorResponse]):
    module: ClassVar[ModuleType]
    check_timeout_s: ClassVar[float] = 120

    async def score(self, data: Example) -> EvaluatorResponse:
        registry = _registry(self.module)
        declared = {
            c.name: c for c in parse_checks(Path(_file(self.module)).read_text())
        }
        case = Case(
            case_id=data["case_id"],
            input=data["input"],
            name=data["name"] if "name" in data else None,
            tags=list(data["tags"]) if "tags" in data else [],
            metadata=dict(data["metadata"]) if "metadata" in data else {},
        )
        trace = Trace(
            id=data["trace_id"],
            project_id=data["project_id"],
            client=JudgmentSyncClient(
                os.environ["JUDGMENT_TESTING_API_URL"],
                os.environ["JUDGMENT_TESTING_TOKEN"],
                data["organization_id"],
            ),
        )
        results = [
            await self._run(name, registry.get(name), declared.get(name), case, trace)
            for name in data["checks"]
        ]
        gating = [
            r
            for r in results
            if r["check_name"] in declared and declared[r["check_name"]].gating
        ]
        ok = all(r["passed"] is True for r in gating)
        return EvaluatorResponse(
            value=ok,
            reason=f"{sum(r['passed'] is True for r in gating)}/{len(gating)} gating checks passed",
            results=results,
        )

    async def _run(
        self,
        name: str,
        fn: Optional[CheckFn],
        check: Optional[Check],
        case: Case,
        trace: Trace,
    ) -> dict[str, Any]:
        if fn is None or check is None:
            return _error(name, "not declared in the evaluator")
        args: tuple[Any, ...] = (case, trace)
        if len(inspect.signature(fn).parameters) >= 3:
            args += (case.metadata.get("checks", {}).get(name),)
        try:
            if inspect.iscoroutinefunction(fn):
                returned = await asyncio.wait_for(fn(*args), self.check_timeout_s)
            else:
                # ponytail: a timed-out sync check keeps its thread; fine in a single-use container.
                returned = await asyncio.wait_for(
                    asyncio.to_thread(fn, *args), self.check_timeout_s
                )
        except asyncio.TimeoutError:
            return _error(name, f"timed out after {self.check_timeout_s:g}s")
        except Exception as e:
            return _error(name, f"{type(e).__name__}: {e}")
        return _normalize(name, check, returned)


def _normalize(name: str, check: Check, returned: Any) -> dict[str, Any]:
    result = returned if isinstance(returned, Result) else Result(returned)
    value = result.value
    verdict: Optional[bool]
    if check.kind == "pass_fail":
        if not isinstance(value, bool):
            return _error(
                name, f"pass_fail check returned {type(value).__name__}, not bool"
            )
        verdict = value if result.passed is None else result.passed
    elif check.kind == "label":
        if value not in check.label_set:
            return _error(name, f"label {value!r} is not one of {check.label_set}")
        verdict = result.passed
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _error(name, f"metric returned {type(value).__name__}, not a number")
        value = float(value)
        verdict = (
            result.passed if result.passed is not None else _within_budget(check, value)
        )
    return {
        "check_name": name,
        "value": value,
        "passed": verdict,
        "probability": result.probability,
        "reason": result.reason,
        "evidence": result.evidence,
        "cost_usd": None,
        "error": None,
    }


def _within_budget(check: Check, value: float) -> Optional[bool]:
    metric = check.metric or {}
    if "budget" not in metric:
        return None
    budget = float(metric["budget"])
    return value <= budget if metric["better_when"] == "lower" else value >= budget


def _error(name: str, message: str) -> dict[str, Any]:
    return {
        "check_name": name,
        "value": None,
        "passed": False,
        "probability": None,
        "reason": None,
        "evidence": [],
        "cost_usd": None,
        "error": message,
    }


def _registry(module: ModuleType) -> dict[str, CheckFn]:
    found = [v for v in vars(module).values() if isinstance(v, Evaluator)]
    if len(found) != 1:
        raise ValueError(
            f"expected one evaluator() in {module.__name__}, found {len(found)}"
        )
    return found[0].checks


def _file(module: ModuleType) -> str:
    if not module.__file__:
        raise ValueError(f"{module.__name__} has no source file")
    return module.__file__
