from __future__ import annotations

from copy import deepcopy

import pytest

from judgeval.data import (
    Example,
    ScorerData,
    ScoringResult,
    from_openeval_result_set,
    to_openeval_result_set,
    validate_openeval_result_set,
)
from judgeval.data.trace import Trace


def _result_set(results: list[ScoringResult], **kwargs) -> dict:
    return to_openeval_result_set(
        results,
        suite_id="suite-1",
        run_id="run-1",
        started_at="2026-09-07T00:00:00Z",
        **kwargs,
    )


def test_numeric_and_categorical_scorers_round_trip_losslessly():
    example = Example(
        example_id="example-1",
        created_at="2026-09-07T00:00:00Z",
        name="capital",
        _properties={"actual_output": "Paris", "input": "capital of France"},
    )
    original = ScoringResult(
        scorers_data=[
            ScorerData(
                name="accuracy",
                value=75.0,
                score_type="numeric",
                minimum_score_range=0,
                maximum_score_range=100,
                evaluation_model="judge-model",
                additional_metadata={"rubric": "v2"},
                id="accuracy-1",
                success=True,
            ),
            ScorerData(
                name="label",
                value="gold",
                score_type="categorical",
                additional_metadata={"labels": ["silver", "gold"]},
                id="label-1",
                success=True,
            ),
        ],
        data_object=example,
        name="nightly",
        trace_id="trace-1",
        run_duration=1.234,
        evaluation_cost=0.002,
    )

    result_set = _result_set([original])
    validate_openeval_result_set(result_set)
    record = result_set["results"][0]
    assert record["test_case_id"] == "example-1"
    assert record["actual_output"] == "Paris"
    assert record["duration_ms"] == 1234
    numeric, categorical = record["grader_results"]
    assert numeric["score"] == 0.75
    assert numeric["passed"] is True
    assert numeric["metadata"]["openeval.raw_score"] == 75.0
    assert numeric["type"] == "llm_judge"
    # Rule 6: a categorical scorer has no numeric score, so its grader is
    # "not verified" and must not pass even though the row verdict is True.
    assert categorical["score"] is None
    assert categorical["passed"] is False
    # the row verdict reports on the record, not on the unverifiable grader
    assert record["passed"] is True

    restored = from_openeval_result_set(result_set)
    assert restored == [original]


def test_row_verdict_lives_on_the_record_not_on_each_grader():
    example = Example(example_id="example-2", created_at="2026-09-07T00:00:00Z")
    original = ScoringResult(
        scorers_data=[
            # a full-marks scorer on a row whose pass condition failed: the
            # grader verified its own outcome, the row verdict is false
            ScorerData(
                name="accuracy",
                value=5.0,
                score_type="numeric",
                id="a-1",
                success=False,
            ),
            ScorerData(
                name="verdict", value="No", score_type="binary", id="b-1", success=False
            ),
        ],
        data_object=example,
    )

    result_set = _result_set([original])
    record = result_set["results"][0]
    numeric, binary = record["grader_results"]
    assert numeric["score"] == 1.0
    assert numeric["passed"] is True
    assert binary["score"] == 0.0
    assert binary["passed"] is False
    assert record["passed"] is False

    assert from_openeval_result_set(result_set) == [original]


def test_unscored_rows_and_empty_scorers_are_not_passes():
    example = Example(example_id="example-3", created_at="2026-09-07T00:00:00Z")
    unscored = ScoringResult(
        scorers_data=[
            ScorerData(
                name="label", value="gold", score_type="categorical", success=True
            )
        ],
        data_object=example,
    )
    empty = ScoringResult(scorers_data=[], data_object=deepcopy(example))

    result_set = _result_set([unscored, empty])
    assert result_set["results"][0]["passed"] is False
    assert result_set["results"][0]["grader_results"][0]["passed"] is False
    assert result_set["results"][1]["passed"] is False
    assert from_openeval_result_set(result_set) == [unscored, empty]


def test_errored_numeric_scorer_exports_instead_of_raising():
    example = Example(example_id="example-4", created_at="2026-09-07T00:00:00Z")
    original = ScoringResult(
        scorers_data=[
            ScorerData(
                name="faithfulness",
                score_type="numeric",
                error="judge call timed out",
                success=True,
            )
        ],
        data_object=example,
    )

    result_set = _result_set([original])
    grader = result_set["results"][0]["grader_results"][0]
    assert grader["score"] is None
    assert grader["passed"] is False
    assert grader["metadata"]["error"] == "judge call timed out"
    assert result_set["results"][0]["passed"] is False
    assert from_openeval_result_set(result_set) == [original]


def test_grader_id_is_stable_across_results():
    example = Example(example_id="example-5", created_at="2026-09-07T00:00:00Z")
    results = [
        ScoringResult(
            scorers_data=[
                ScorerData(name="accuracy", value=1.0, score_type="numeric", id="row-1")
            ],
            data_object=example,
        ),
        ScoringResult(
            scorers_data=[
                ScorerData(name="accuracy", value=0.5, score_type="numeric", id="row-2")
            ],
            data_object=deepcopy(example),
        ),
    ]

    result_set = _result_set(results)
    grader_ids = [r["grader_results"][0]["grader_id"] for r in result_set["results"]]
    assert grader_ids == ["accuracy", "accuracy"]


def test_example_trace_and_reserved_name_properties_round_trip():
    span = {
        "organization_id": "org-1",
        "project_id": "project-1",
        "user_id": "user-1",
        "timestamp": "2026-09-07T00:00:00Z",
        "trace_id": "trace-1",
        "span_id": "span-1",
        "resource_attributes": {},
        "span_attributes": {"model": "judge"},
        "duration": "1s",
        "status_code": 0.0,
        "events": [],
    }
    original = ScoringResult(
        scorers_data=[
            ScorerData(name="binary", value="Yes", score_type="binary", success=True)
        ],
        data_object=Example(
            example_id="real-id",
            created_at="2026-09-07T00:00:00Z",
            name="real-name",
            _properties={
                "input": "q",
                "example_id": "shadow",
                "name": "user-prop",
                "created_at": "user-date",
            },
            trace=Trace(spans=[span]),
        ),
    )

    result_set = _result_set([original])
    restored = from_openeval_result_set(result_set)
    example = restored[0].data_object
    assert example.example_id == "real-id"
    assert example.name == "real-name"
    assert example.created_at == "2026-09-07T00:00:00Z"
    assert example.properties == {
        "input": "q",
        "example_id": "shadow",
        "name": "user-prop",
        "created_at": "user-date",
    }
    assert example.trace is not None
    assert example.trace.spans[0]["span_id"] == "span-1"
    assert restored == [original]


def test_trace_span_round_trips_as_a_trace_span():
    span = {
        "organization_id": "org-1",
        "project_id": "project-1",
        "user_id": "user-1",
        "timestamp": "2026-09-07T00:00:00Z",
        "trace_id": "trace-1",
        "span_id": "span-1",
        "resource_attributes": {},
        "span_attributes": {"model": "judge"},
        "duration": "1s",
        "status_code": 0.0,
        "events": [],
    }
    original = ScoringResult(
        scorers_data=[
            ScorerData(name="binary", value="Yes", score_type="binary", success=True)
        ],
        data_object=span,
    )

    result_set = _result_set([original])
    assert result_set["results"][0]["test_case_id"] == "span-1"
    assert result_set["results"][0]["grader_results"][0]["score"] == 1.0
    restored = from_openeval_result_set(result_set)
    assert restored == [original]
    assert isinstance(restored[0].data_object, dict)


@pytest.mark.parametrize(
    "timestamp",
    ["2026-09-07", "2026-09-07T00:00:00", "2026-09-07 00:00:00Z"],
)
def test_non_rfc3339_timestamps_are_rejected(timestamp):
    example = Example(example_id="example-6", created_at="2026-09-07T00:00:00Z")
    results = [
        ScoringResult(
            scorers_data=[ScorerData(name="score", value=0.5, score_type="numeric")],
            data_object=example,
        )
    ]

    with pytest.raises(ValueError, match="RFC 3339"):
        to_openeval_result_set(
            results,
            suite_id="suite-1",
            run_id="run-1",
            started_at=timestamp,
        )


@pytest.mark.parametrize(
    "timestamp", ["2026-09-07T00:00:00Z", "2026-09-07T00:00:00+02:00"]
)
def test_rfc3339_timestamps_with_offsets_are_accepted(timestamp):
    example = Example(example_id="example-7", created_at=timestamp)
    results = [
        ScoringResult(
            scorers_data=[ScorerData(name="score", value=0.5, score_type="numeric")],
            data_object=example,
        )
    ]

    result_set = to_openeval_result_set(
        results,
        suite_id="suite-1",
        run_id="run-1",
        started_at=timestamp,
        completed_at=timestamp,
    )
    assert result_set["completed_at"] == timestamp


def test_completed_at_is_omitted_when_not_supplied():
    example = Example(example_id="example-8", created_at="2026-09-07T00:00:00Z")
    results = [
        ScoringResult(
            scorers_data=[ScorerData(name="score", value=0.5, score_type="numeric")],
            data_object=example,
        )
    ]

    result_set = _result_set(results)
    assert "completed_at" not in result_set
    validate_openeval_result_set(result_set)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda result_set: result_set.pop("run_id"), "missing or unsupported fields"),
        (lambda result_set: result_set.update(unexpected=True), "unsupported fields"),
        (
            lambda result_set: result_set["results"][0]["grader_results"][0].update(
                score=1.1
            ),
            "number from 0 to 1",
        ),
        (
            lambda result_set: result_set["results"][0]["grader_results"][0].update(
                passed=True
            ),
            "passed must be false when score is null",
        ),
        (
            lambda result_set: result_set["results"][0].update(passed=True),
            "passed must be false when no grader produced a score",
        ),
    ],
)
def test_rejects_malformed_result_sets(mutate, message):
    original = ScoringResult(
        scorers_data=[
            ScorerData(
                name="label", value="gold", score_type="categorical", success=True
            )
        ],
        data_object=Example(example_id="example-9", created_at="2026-09-07T00:00:00Z"),
    )
    malformed = deepcopy(_result_set([original]))
    mutate(malformed)

    with pytest.raises(ValueError, match=message):
        from_openeval_result_set(malformed)


def test_rejects_result_sets_without_lossless_judgeval_metadata():
    original = ScoringResult(
        scorers_data=[ScorerData(name="score", value=0.5, score_type="numeric")],
        data_object=Example(example_id="example-10", created_at="2026-09-07T00:00:00Z"),
    )
    result_set = _result_set([original])
    del result_set["results"][0]["metadata"]

    with pytest.raises(ValueError, match="missing or unsupported fields"):
        from_openeval_result_set(result_set)
