from __future__ import annotations

from unittest.mock import MagicMock
import json
import httpx

import pytest

from judgeval.exceptions import (
    JudgmentAPIError,
    JudgmentConflictError,
    JudgmentValidationError,
)
from judgeval.external_judges.external_judge import ExternalJudge
from judgeval.external_judges.external_judge_factory import ExternalJudgeFactory
from judgeval.internal.api import JudgmentSyncClient


def _make_factory(project_id="proj-1"):
    client = MagicMock()
    return ExternalJudgeFactory(
        client=client, project_id=project_id, project_name="test"
    ), client


class TestExternalJudgeFactoryCreate:
    def test_create_returns_external_judge(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {
            "judge_id": "judge-1",
            "judge_version_id": "version-1",
            "implementation_id": None,
        }
        result = factory.create(name="human-thumbs-up", score_type="binary")
        assert isinstance(result, ExternalJudge)
        assert result.judge_id == "judge-1"
        assert result.judge_version_id == "version-1"
        assert result.name == "human-thumbs-up"
        assert result.score_type == "binary"

    def test_create_binary_payload_shape(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {
            "judge_id": "judge-1",
            "judge_version_id": "version-1",
            "implementation_id": None,
        }
        factory.create(name="n", score_type="binary")
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload == {
            "judge": {
                "name": "n",
                "executionMode": "external",
                "initialVersion": {"scoreType": "binary"},
            },
        }

    def test_create_includes_judge_description_when_provided(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {
            "judge_id": "judge-1",
            "judge_version_id": "version-1",
            "implementation_id": None,
        }
        factory.create(name="n", score_type="numeric", judge_description="d")
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload["judge"]["judgeDescription"] == "d"

    def test_create_categorical_includes_outputs(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {
            "judge_id": "judge-1",
            "judge_version_id": "version-1",
            "implementation_id": None,
        }
        outputs = [
            {"name": "good", "description": "Good response"},
            {"name": "bad", "description": "Bad response"},
        ]
        factory.create(name="n", score_type="categorical", outputs=outputs)
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload["judge"]["initialVersion"] == {
            "scoreType": "categorical",
            "outputs": outputs,
        }

    def test_create_categorical_without_outputs_raises(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError, match="outputs"):
            factory.create(name="n", score_type="categorical")

    def test_create_categorical_with_one_output_raises(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError, match="outputs"):
            factory.create(
                name="n",
                score_type="categorical",
                outputs=[{"name": "good", "description": ""}],
            )

    def test_create_binary_with_outputs_raises(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError, match="categorical"):
            factory.create(
                name="n",
                score_type="binary",
                outputs=[{"name": "good", "description": ""}],
            )

    def test_create_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.create(name="n", score_type="binary")
        assert result is None

    def test_create_conflict_maps_to_conflict_error(self):
        factory, client = _make_factory()
        client.post_projects_judges.side_effect = JudgmentAPIError(
            409, "a judge with this name already exists", None
        )
        with pytest.raises(JudgmentConflictError):
            factory.create(name="dupe", score_type="binary")

    def test_create_validation_maps_to_validation_error(self):
        factory, client = _make_factory()
        client.post_projects_judges.side_effect = JudgmentAPIError(
            422, "invalid judge configuration", None
        )
        with pytest.raises(JudgmentValidationError):
            factory.create(name="bad", score_type="binary")


class TestExternalJudgeFactorySubmitResult:
    def test_submit_result_by_judge_id_returns_id(self):
        factory, client = _make_factory()
        client.post_projects_judge_results.return_value = {"id": "result-1"}
        result = factory.submit_result(
            judge_id="judge-1",
            target={"type": "trace", "trace_id": "trace-1"},
            value=True,
        )
        assert result == "result-1"

    def test_submit_result_requires_exactly_one_judge_identifier(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError, match="Exactly one"):
            factory.submit_result(
                target={"type": "trace", "trace_id": "trace-1"}, value=True
            )

    def test_submit_result_rejects_both_judge_identifiers(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError, match="Exactly one"):
            factory.submit_result(
                judge_id="judge-1",
                judge_name="human-thumbs-up",
                target={"type": "trace", "trace_id": "trace-1"},
                value=True,
            )

    def test_submit_result_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.submit_result(
            judge_id="judge-1",
            target={"type": "trace", "trace_id": "trace-1"},
            value=True,
        )
        assert result is None

    def test_submit_result_validation_maps_to_validation_error(self):
        factory, client = _make_factory()
        client.post_projects_judge_results.side_effect = JudgmentAPIError(
            422, "value does not match judge score type", None
        )
        with pytest.raises(JudgmentValidationError):
            factory.submit_result(
                judge_id="judge-1",
                target={"type": "trace", "trace_id": "trace-1"},
                value=True,
            )

    def test_submit_result_not_found_raises_raw_api_error(self):
        factory, client = _make_factory()
        client.post_projects_judge_results.side_effect = JudgmentAPIError(
            404, "trace not found", None
        )
        with pytest.raises(JudgmentAPIError):
            factory.submit_result(
                judge_id="judge-1",
                target={"type": "trace", "trace_id": "trace-1"},
                value=True,
            )


def test_session_score_can_be_resubmitted_without_a_trace():
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "judge_id": "judge-1",
                "judge_version_id": "version-1",
                "implementation_id": None,
            }
            if request.url.path.endswith("/judges")
            else {"id": "result-1"},
        )

    client = JudgmentSyncClient("https://api.test", "key", "org")
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    factory = ExternalJudgeFactory(client, "proj-1", "test")
    judge = factory.create(name="rating", score_type="numeric")
    result = factory.submit_result(
        judge_id="judge-1",
        target={"type": "session", "session_id": "session-1"},
        value=4.5,
        idempotency_key="fa000000-0000-0000-0000-000000000011",
    )
    updated = factory.submit_result(
        judge_id="judge-1",
        target={"type": "session", "session_id": "session-1"},
        value=3,
        idempotency_key="fa000000-0000-0000-0000-000000000012",
    )
    assert (judge, result, updated, requests) == (
        ExternalJudge("judge-1", "version-1", "rating", "numeric"),
        "result-1",
        "result-1",
        [
            (
                "POST",
                "/v1/projects/proj-1/judges",
                {
                    "judge": {
                        "name": "rating",
                        "executionMode": "external",
                        "initialVersion": {"scoreType": "numeric"},
                    }
                },
            ),
            (
                "POST",
                "/v1/projects/proj-1/judge-results",
                {
                    "judge_id": "judge-1",
                    "target": {"type": "session", "session_id": "session-1"},
                    "value": 4.5,
                    "idempotency_key": "fa000000-0000-0000-0000-000000000011",
                },
            ),
            (
                "POST",
                "/v1/projects/proj-1/judge-results",
                {
                    "judge_id": "judge-1",
                    "target": {"type": "session", "session_id": "session-1"},
                    "value": 3,
                    "idempotency_key": "fa000000-0000-0000-0000-000000000012",
                },
            ),
        ],
    )


def test_rejects_unknown_score_types_instead_of_creating_numeric_judges():
    factory, _ = _make_factory()
    with pytest.raises(ValueError, match="score_type"):
        factory.create(name="feedback", score_type="numerical")


def test_created_names_can_be_used_to_submit_categorical_feedback():
    stored = {}

    def handler(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/judges"):
            judge = payload["judge"]
            stored["name"] = judge["name"].strip()
            stored["outputs"] = [
                {**output, "name": output["name"].strip()}
                for output in judge["initialVersion"]["outputs"]
            ]
            return httpx.Response(
                200,
                json={
                    "judge_id": "j",
                    "judge_version_id": "v",
                    "implementation_id": None,
                },
            )
        if (
            payload["judge_name"] != stored["name"]
            or payload["value"] != stored["outputs"][0]["name"]
        ):
            return httpx.Response(404, json={"message": "Judge or output not found"})
        return httpx.Response(200, json={"id": "feedback-1"})

    client = JudgmentSyncClient("https://api.test", "key", "org")
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    factory = ExternalJudgeFactory(client, "project", "test")
    judge = factory.create(
        name=" topic ",
        score_type="categorical",
        outputs=[
            {"name": " good ", "description": "Good"},
            {"name": " bad ", "description": "Bad"},
        ],
    )
    assert judge is not None and judge.outputs is not None
    result = factory.submit_result(
        judge_name=judge.name,
        target={"type": "trace", "trace_id": "trace"},
        value=judge.outputs[0]["name"],
    )
    assert (judge.name, judge.outputs, result) == (
        "topic",
        [
            {"name": "good", "description": "Good"},
            {"name": "bad", "description": "Bad"},
        ],
        "feedback-1",
    )


def test_rejects_ambiguous_targets_before_sending_feedback():
    factory, _ = _make_factory()
    with pytest.raises(ValueError, match="target"):
        factory.submit_result(
            judge_id="judge-1",
            target={"type": "session", "session_id": "session", "trace_id": "trace"},
            value=True,
        )


def test_retrying_a_correction_reuses_its_submission_identity():
    factory, client = _make_factory()
    request_key = "fa000000-0000-0000-0000-000000000001"
    result_id = "fa000000-0000-0000-0000-000000000002"
    accepted = {}

    def accept(*, project_id, payload):
        assert project_id == "proj-1"
        accepted[payload["idempotency_key"]] = payload["result_id"]
        return {"id": accepted[payload["idempotency_key"]]}

    client.post_projects_judge_results.side_effect = accept
    first = factory.submit_result(
        judge_id="judge-1",
        target={"type": "session", "session_id": "session-1"},
        value=False,
        idempotency_key=request_key,
        result_id=result_id,
    )
    retry = factory.submit_result(
        judge_id="judge-1",
        target={"type": "session", "session_id": "session-1"},
        value=False,
        idempotency_key=request_key,
        result_id=result_id,
    )
    assert (first, retry, accepted) == (result_id, result_id, {request_key: result_id})
