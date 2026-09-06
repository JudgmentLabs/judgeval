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


def _result_set(results: list[ScoringResult]) -> dict:
    return to_openeval_result_set(
        results,
        suite_id="suite-1",
        run_id="run-1",
        started_at="2026-09-07T00:00:00Z",
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
                error="transient note",
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
                success=None,
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
    assert record["grader_results"][0]["score"] == 0.75
    assert record["grader_results"][0]["passed"] is True
    assert record["grader_results"][1]["score"] is None
    assert record["grader_results"][1]["passed"] is False

    restored = from_openeval_result_set(result_set)
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
    ],
)
def test_rejects_malformed_result_sets(mutate, message):
    original = ScoringResult(
        scorers_data=[ScorerData(name="score", value=0.5, score_type="numeric")],
        data_object=Example(example_id="example-1", created_at="2026-09-07T00:00:00Z"),
    )
    malformed = deepcopy(_result_set([original]))
    mutate(malformed)

    with pytest.raises(ValueError, match=message):
        from_openeval_result_set(malformed)


def test_rejects_result_sets_without_lossless_judgeval_metadata():
    original = ScoringResult(
        scorers_data=[ScorerData(name="score", value=0.5, score_type="numeric")],
        data_object=Example(example_id="example-1", created_at="2026-09-07T00:00:00Z"),
    )
    result_set = _result_set([original])
    del result_set["results"][0]["metadata"]

    with pytest.raises(ValueError, match="missing or unsupported fields"):
        from_openeval_result_set(result_set)
