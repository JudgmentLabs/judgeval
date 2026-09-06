"""Optional conversion between Judgeval results and EvalPort ResultSets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from numbers import Real
import re
from typing import Any, Dict, List, cast

from judgeval.data.example import Example
from judgeval.data.scorer_data import ScorerData
from judgeval.data.scoring_result import ScoringResult
from judgeval.internal.api.models import TraceSpan

OPENEVAL_VERSION = "1.0.0"
_METADATA_NAMESPACE = "judgeval"
_RESULT_SET_KEYS = {
    "version",
    "suite_id",
    "run_id",
    "started_at",
    "completed_at",
    "runner",
    "results",
    "summary",
}
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _timestamp(value: Any, path: str) -> None:
    _require(isinstance(value, str), f"{path} must be a string")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{path} must be an ISO 8601 timestamp") from error


def _score(scorer: ScorerData) -> float | None:
    if scorer.score_type == "numeric":
        _require(
            not isinstance(scorer.value, bool) and isinstance(scorer.value, Real),
            f"numeric scorer {scorer.name!r} must have a numeric value",
        )
        _require(
            scorer.minimum_score_range < scorer.maximum_score_range,
            f"numeric scorer {scorer.name!r} must have minimum < maximum",
        )
        normalized = (float(cast(Real, scorer.value)) - scorer.minimum_score_range) / (
            scorer.maximum_score_range - scorer.minimum_score_range
        )
        return min(1.0, max(0.0, normalized))
    if scorer.score_type == "binary" and isinstance(scorer.value, str):
        return {"yes": 1.0, "no": 0.0}.get(scorer.value.lower())
    return None


def _data_object_metadata(data_object: Example | TraceSpan) -> Dict[str, Any]:
    if isinstance(data_object, Example):
        return {"kind": "example", "value": data_object.to_dict()}
    return {"kind": "trace_span", "value": dict(data_object)}


def _grader_result(scorer: ScorerData) -> Dict[str, Any]:
    return {
        "grader_id": scorer.id or scorer.name,
        "type": scorer.score_type or "custom",
        "score": _score(scorer),
        # EvalPort requires a boolean. The original nullable value remains in
        # metadata so importing an exported ResultSet is lossless.
        "passed": scorer.success if scorer.success is not None else False,
        "metadata": {_METADATA_NAMESPACE: {"scorer_data": scorer.to_dict()}},
    }


def _result_record(result: ScoringResult) -> Dict[str, Any]:
    data_object = result.data_object
    record: Dict[str, Any] = {
        "test_case_id": data_object.example_id
        if isinstance(data_object, Example)
        else data_object["span_id"],
        "grader_results": [_grader_result(scorer) for scorer in result.scorers_data],
        "passed": all(scorer.success is True for scorer in result.scorers_data),
        "metadata": {
            _METADATA_NAMESPACE: {
                "data_object": _data_object_metadata(data_object),
                "name": result.name,
                "trace_id": result.trace_id,
                "run_duration": result.run_duration,
                "evaluation_cost": result.evaluation_cost,
            }
        },
    }
    if isinstance(data_object, Example):
        actual_output = data_object.properties.get("actual_output")
        if isinstance(actual_output, str):
            record["actual_output"] = actual_output
    if result.run_duration is not None:
        _require(result.run_duration >= 0, "run_duration must not be negative")
        record["duration_ms"] = round(result.run_duration * 1000)
    return record


def to_openeval_result_set(
    results: Sequence[ScoringResult],
    *,
    suite_id: str,
    run_id: str,
    started_at: str,
    completed_at: str | None = None,
) -> Dict[str, Any]:
    """Export Judgeval results to the portable EvalPort ResultSet shape.

    Numeric values are normalized from their declared range, binary values map
    to 1 or 0, and categorical values use ``null`` because EvalPort scores are
    numeric. Raw Judgeval data stays in ``metadata["judgeval"]`` for import.
    """
    records = [_result_record(result) for result in results]
    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at or started_at,
        "runner": {"name": "judgeval"},
        "results": records,
        "summary": {
            "total": len(records),
            "passed": sum(record["passed"] for record in records),
            "failed": sum(not record["passed"] for record in records),
            "pass_rate": sum(record["passed"] for record in records) / len(records)
            if records
            else 0,
        },
    }
    validate_openeval_result_set(result_set)
    return result_set


def validate_openeval_result_set(result_set: Mapping[str, Any]) -> None:
    """Validate the strict EvalPort-compatible shape this adapter emits.

    It validates this adapter's lossless envelope rather than importing EvalPort
    or reimplementing every optional field in its schema.
    """
    _require(
        set(result_set) == _RESULT_SET_KEYS,
        "result_set has missing or unsupported fields",
    )
    version = result_set["version"]
    _require(
        isinstance(version, str) and bool(_SEMVER_PATTERN.fullmatch(version)),
        "result_set.version must be a semantic version",
    )
    for field in ("suite_id", "run_id"):
        _require(
            isinstance(result_set[field], str) and bool(result_set[field]),
            f"result_set.{field} must be a non-empty string",
        )
    _timestamp(result_set["started_at"], "result_set.started_at")
    _timestamp(result_set["completed_at"], "result_set.completed_at")
    _require(
        result_set["runner"] == {"name": "judgeval"},
        "result_set.runner must identify judgeval",
    )
    results = result_set["results"]
    _require(
        isinstance(results, list) and bool(results),
        "result_set.results must be a non-empty array",
    )
    for index, record in enumerate(results):
        _validate_record(record, f"result_set.results[{index}]")
    summary = result_set["summary"]
    _require(
        isinstance(summary, Mapping)
        and set(summary) == {"total", "passed", "failed", "pass_rate"},
        "result_set.summary has missing or unsupported fields",
    )
    for field in ("total", "passed", "failed"):
        _require(
            isinstance(summary[field], int)
            and not isinstance(summary[field], bool)
            and summary[field] >= 0,
            f"result_set.summary.{field} must be a non-negative integer",
        )
    _require(
        isinstance(summary["pass_rate"], Real)
        and not isinstance(summary["pass_rate"], bool)
        and 0 <= float(summary["pass_rate"]) <= 1,
        "result_set.summary.pass_rate must be a number from 0 to 1",
    )


def _validate_record(value: Any, path: str) -> None:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    expected = {"test_case_id", "grader_results", "passed", "metadata"}
    allowed = expected | {"actual_output", "duration_ms"}
    _require(
        expected <= value.keys() <= allowed, f"{path} has missing or unsupported fields"
    )
    _require(
        isinstance(value["test_case_id"], str) and bool(value["test_case_id"]),
        f"{path}.test_case_id must be a non-empty string",
    )
    _require(isinstance(value["passed"], bool), f"{path}.passed must be a boolean")
    if "actual_output" in value:
        _require(
            isinstance(value["actual_output"], str),
            f"{path}.actual_output must be a string",
        )
    if "duration_ms" in value:
        _require(
            isinstance(value["duration_ms"], int)
            and not isinstance(value["duration_ms"], bool)
            and value["duration_ms"] >= 0,
            f"{path}.duration_ms must be a non-negative integer",
        )
    metadata = value["metadata"]
    _require(
        isinstance(metadata, Mapping) and _METADATA_NAMESPACE in metadata,
        f"{path}.metadata must contain Judgeval round-trip data",
    )
    graders = value["grader_results"]
    _require(isinstance(graders, list), f"{path}.grader_results must be an array")
    for index, grader in enumerate(graders):
        _validate_grader(grader, f"{path}.grader_results[{index}]")


def _validate_grader(value: Any, path: str) -> None:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"grader_id", "type", "score", "passed", "metadata"},
        f"{path} has missing or unsupported fields",
    )
    _require(
        isinstance(value["grader_id"], str) and bool(value["grader_id"]),
        f"{path}.grader_id must be a non-empty string",
    )
    _require(isinstance(value["type"], str), f"{path}.type must be a string")
    score = value["score"]
    _require(
        score is None
        or (
            isinstance(score, Real)
            and not isinstance(score, bool)
            and 0 <= float(score) <= 1
        ),
        f"{path}.score must be a number from 0 to 1 or null",
    )
    _require(isinstance(value["passed"], bool), f"{path}.passed must be a boolean")
    metadata = value["metadata"]
    _require(
        isinstance(metadata, Mapping) and _METADATA_NAMESPACE in metadata,
        f"{path}.metadata must contain Judgeval round-trip data",
    )


def _restore_data_object(value: Any) -> Example | TraceSpan:
    _require(
        isinstance(value, Mapping), "metadata.judgeval.data_object must be an object"
    )
    serialized = value.get("value")
    _require(
        isinstance(serialized, Mapping),
        "metadata.judgeval.data_object.value must be an object",
    )
    if value.get("kind") == "trace_span":
        return cast(TraceSpan, dict(serialized))
    _require(value.get("kind") == "example", "unsupported Judgeval data object kind")
    example_id = serialized.get("example_id")
    created_at = serialized.get("created_at")
    _require(
        isinstance(example_id, str) and isinstance(created_at, str),
        "Example metadata is incomplete",
    )
    name = serialized.get("name")
    _require(
        name is None or isinstance(name, str), "Example.name must be a string or null"
    )
    example = Example(example_id=example_id, created_at=created_at, name=name)
    for key, item in serialized.items():
        if key not in {"example_id", "created_at", "name"}:
            example._properties[key] = item
    return example


def from_openeval_result_set(result_set: Mapping[str, Any]) -> List[ScoringResult]:
    """Restore Judgeval results from a ResultSet emitted by this adapter."""
    validate_openeval_result_set(result_set)
    restored: List[ScoringResult] = []
    for record in result_set["results"]:
        metadata = cast(Mapping[str, Any], record["metadata"])[_METADATA_NAMESPACE]
        _require(
            isinstance(metadata, Mapping), "Judgeval result metadata must be an object"
        )
        scorers = []
        for grader in record["grader_results"]:
            scorer_metadata = cast(Mapping[str, Any], grader["metadata"])[
                _METADATA_NAMESPACE
            ]
            _require(
                isinstance(scorer_metadata, Mapping),
                "Judgeval scorer metadata must be an object",
            )
            scorer_data = scorer_metadata.get("scorer_data")
            _require(
                isinstance(scorer_data, Mapping),
                "Judgeval scorer metadata is incomplete",
            )
            scorers.append(ScorerData(**scorer_data))
        restored.append(
            ScoringResult(
                scorers_data=scorers,
                data_object=_restore_data_object(metadata.get("data_object")),
                name=metadata.get("name"),
                trace_id=metadata.get("trace_id"),
                run_duration=metadata.get("run_duration"),
                evaluation_cost=metadata.get("evaluation_cost"),
            )
        )
    return restored
