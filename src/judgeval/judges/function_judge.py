"""Function judges: one Python file defining `judge(example)`, run in Monty.

The local test runs a judge the way Judgment's code-judge runner does, with the same
Monty limits, upload check and result check, so a judge that passes here
behaves the same once uploaded.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from judgeval.api import JudgmentApiClient
from judgeval.internal.api.models import CreateJudgeRequestJudge1, UpdateJudgeRequest

OutputType = Literal["binary", "numeric", "categorical"]

# Keep in step with judgment-mono services/code-judge-runner/src/sandbox.ts:
# the limits, type-check stubs, judge-call check and result rules must match.
_LIMITS = {
    "max_memory": 256 * 1024 * 1024,
    "max_feed_duration_secs": 10,
    "max_total_sleep_secs": 0,
}
_REQUEST_TIMEOUT_SECONDS = 30
# The upload check: type-check against the one global a judge gets, and fail
# if `judge(example)` is missing or has the wrong shape.
_TYPE_CHECK_STUBS = "from typing import Any\nexample: dict[str, Any]\n"
_JUDGE_CALL_CHECK = "\ndef _check_judge_call():\n    judge(example)\n"
_RESULT_KEYS = {"value", "reason", "citations", "metrics"}
_MAX_METRICS = 50
_MAX_RESULT_BYTES = 256 * 1024


class JudgeFailed(Exception):
    """The judge itself failed: it raised, hit a limit or returned a bad result."""


def is_function_judge(source: str) -> bool:
    """Whether the file defines a top-level `judge` function."""
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "judge" for node in tree.body
    )


def _monty() -> Any:
    try:
        import pydantic_monty
    except ImportError as error:
        raise RuntimeError(
            "Running a function judge needs Monty: pip install 'judgeval[monty]'"
        ) from error
    return pydantic_monty


def _print_to_stderr(_stream: str, text: str) -> None:
    sys.stderr.write(text)


def source_problems(source: str) -> Optional[str]:
    """The upload check: each problem with its line, or None when Monty can run it."""
    monty = _monty()
    with monty.Monty(max_processes=1, request_timeout=_REQUEST_TIMEOUT_SECONDS) as pool:
        with pool.checkout(
            script_name="judge.py",
            limits=_LIMITS,
            type_check=True,
            type_check_stubs=_TYPE_CHECK_STUBS,
            type_check_format="concise",
        ) as session:
            try:
                session.feed_run(
                    source + _JUDGE_CALL_CHECK,
                    inputs={"example": {}},
                    print_callback=lambda *_: None,
                )
            except monty.MontyTypingError as error:
                return str(error.display()).strip()
            except monty.MontyRuntimeError as error:
                frames = error.traceback()
                where = f"judge.py:{frames[-1].line}" if frames else "judge.py"
                return f"{where}: {error.display('type-msg')}"
            except monty.MontySyntaxError as error:
                return str(error.display("traceback"))
            except monty.MontyCrashedError as error:
                # The same answer the hosted check gives; any other crash is ours.
                if error.timed_out:
                    return "Module-level code exceeded its time limit"
                raise
    return None


def run_judge(source: str, example: Dict[str, Any]) -> Any:
    """Runs `judge(example)` in Monty and returns what it returned."""
    monty = _monty()
    with monty.Monty(max_processes=1, request_timeout=_REQUEST_TIMEOUT_SECONDS) as pool:
        try:
            with pool.checkout(script_name="judge.py", limits=_LIMITS) as session:
                return session.feed_run(
                    f"{source}\njudge(example)",
                    inputs={"example": example},
                    print_callback=_print_to_stderr,
                )
        except (
            monty.MontyRuntimeError,
            monty.MontySyntaxError,
            monty.MontyTypingError,
        ) as error:
            raise JudgeFailed(error.display("traceback")) from error
        except monty.MontyCrashedError as error:
            if error.timed_out:
                raise JudgeFailed("Judge exceeded its time limit") from error
            raise


def _output_value_problem(
    value: Any, output: OutputType, labels: List[str]
) -> Optional[str]:
    is_number = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
    if output == "binary" and not isinstance(value, bool):
        return "value must be True or False"
    if output == "numeric" and not is_number:
        return "value must be a finite number"
    if output == "categorical" and value not in labels:
        return f"value must be one of {labels}"
    return None


def _output_fields_problem(fields: Dict[str, Any]) -> Optional[str]:
    unknown = sorted(set(fields) - _RESULT_KEYS)
    if unknown:
        return f"unknown keys {unknown}; allowed: value, reason, citations, metrics"
    if "reason" in fields and not isinstance(fields["reason"], str):
        return "reason must be a string"
    citations = fields.get("citations", [])
    if not isinstance(citations, list) or not all(
        isinstance(c, dict)
        and isinstance(c.get("span_id"), str)
        and isinstance(c.get("span_attribute"), str)
        for c in citations
    ):
        return "citations must be a list of {'span_id': str, 'span_attribute': str}"
    metrics = fields.get("metrics", {})
    if not isinstance(metrics, dict) or len(metrics) > _MAX_METRICS:
        return f"metrics must be a dict of at most {_MAX_METRICS} numbers"
    for key, number in metrics.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return f"metrics[{key!r}] must be a number"
        if not abs(number) <= 1e100:
            return f"metrics[{key!r}] must be within ±1e100"
    return None


def check_result(
    returned: Any, output: OutputType, labels: List[str]
) -> Dict[str, Any]:
    """Checks a judge's return value as the code-judge runner does.

    A judge returns its value, or a dict with `value` and optional `reason`,
    `citations` and `metrics`.
    """
    fields = returned if isinstance(returned, dict) else {"value": returned}
    if "value" not in fields:
        raise JudgeFailed("Judge returned an invalid result: missing value")
    problem = _output_value_problem(
        fields["value"], output, labels
    ) or _output_fields_problem(fields)
    if problem:
        raise JudgeFailed(f"Judge returned an invalid result: {problem}")
    # Measured as compact UTF-8 JSON, as the runner's JSON.stringify does.
    size = len(json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode())
    if size > _MAX_RESULT_BYTES:
        raise JudgeFailed(f"Judge returned more than {_MAX_RESULT_BYTES // 1024} KiB")
    return fields


def load_rows(path: Path, key: str) -> List[Dict[str, Any]]:
    """Reads saved records: a JSON list, {key: [...]}, or NDJSON.

    Spans for a trace (key "spans"), or a conversation's log (key
    "conversation"): each model call and item, item content as objects.
    """
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if (
        isinstance(data, dict)
        and key not in data
        and len(text.strip().splitlines()) == 1
    ):
        return [data]  # NDJSON with a single record
    rows = data.get(key) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path} must hold a list or {{'{key}': [...]}}")
    return rows


def upload_function_judge(
    client: JudgmentApiClient,
    project_id: str,
    name: str,
    source: str,
    categories: List[str],
    bump_major: bool,
    conversation: bool = False,
) -> str:
    """Creates the judge, or a new production version of an existing one.

    A new judge is categorical and needs at least two categories, and grades
    whole conversations when `conversation` is set. A new version keeps the
    judge's score type, outputs and what it grades; `categories` are ignored,
    and `conversation` is refused because what a judge grades can't change.
    """
    listed = client.list_judges(project_id)["judges"]
    existing = next((judge for judge in listed if judge["name"] == name), None)

    if existing is None:
        if len(categories) < 2:
            raise ValueError(
                "A new function judge needs at least two categories, e.g. "
                "--category Pass --category Fail"
            )
        judge: CreateJudgeRequestJudge1 = {
            "name": name,
            "judgeType": "custom",
            "initialVersion": {
                "scoreType": "categorical",
                "outputs": [{"name": c, "description": ""} for c in categories],
            },
            "config": {"functionSource": source},
        }
        if conversation:
            judge["conversation"] = True
        client.create_judge(project_id, {"judge": judge})
        return f"Created function judge '{name}'"

    if existing["judge_type"] != "custom":
        raise ValueError(f"Judge '{name}' exists and is not a Code judge")
    if conversation:
        raise ValueError(
            f"Judge '{name}' exists; --conversation only applies when creating "
            "a judge, and what a judge grades can't change"
        )
    # The list endpoint types versions as numbers; the update needs integers.
    major = int(existing["judge_major_version"])
    minor = int(existing["judge_minor_version"])
    target_major, target_minor = (major + 1, 0) if bump_major else (major, minor + 1)
    update: UpdateJudgeRequest = {
        "update": {
            "newVersionFromSource": {
                "sourceVersion": {"major": major, "minor": minor},
                "targetVersion": {"major": target_major, "minor": target_minor},
                "promoteToProduction": True,
                "config": {"functionSource": source},
            }
        }
    }
    client.update_judge(project_id, existing["id"], update)
    return f"Uploaded '{name}' as version {target_major}.{target_minor}"
