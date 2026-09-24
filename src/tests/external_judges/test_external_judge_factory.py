from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from judgeval.exceptions import (
    JudgmentAPIError,
    JudgmentConflictError,
    JudgmentValidationError,
)
from judgeval.external_judges.external_judge import ExternalJudge
from judgeval.external_judges.external_judge_factory import ExternalJudgeFactory


def _make_factory(project_id="proj-1"):
    client = MagicMock()
    return ExternalJudgeFactory(
        client=client, project_id=project_id, project_name="test"
    ), client


class TestExternalJudgeFactoryCreate:
    def test_create_returns_external_judge(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {
            "judgeId": "judge-1",
            "judgeVersionId": "version-1",
            "implementationId": None,
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
            "judgeId": "judge-1",
            "judgeVersionId": "version-1",
            "implementationId": None,
        }
        factory.create(name="n", score_type="binary")
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload == {
            "name": "n",
            "judgeType": "external",
            "initialVersion": {"scoreType": "binary"},
        }

    def test_create_includes_judge_description_when_provided(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {
            "judgeId": "judge-1",
            "judgeVersionId": "version-1",
            "implementationId": None,
        }
        factory.create(name="n", score_type="numeric", judge_description="d")
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload["judgeDescription"] == "d"

    def test_create_categorical_includes_outputs(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {
            "judgeId": "judge-1",
            "judgeVersionId": "version-1",
            "implementationId": None,
        }
        outputs = [
            {"name": "good", "description": "Good response"},
            {"name": "bad", "description": "Bad response"},
        ]
        factory.create(name="n", score_type="categorical", outputs=outputs)
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload["initialVersion"] == {
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
            judge_id="judge-1", trace_id="trace-1", value=True
        )
        assert result == "result-1"

    def test_submit_result_payload_shape(self):
        factory, client = _make_factory()
        client.post_projects_judge_results.return_value = {"id": "result-1"}
        factory.submit_result(
            judge_name="human-thumbs-up",
            trace_id="trace-1",
            value=True,
            session_id="session-1",
            reason="Looked correct.",
        )
        payload = client.post_projects_judge_results.call_args.kwargs["payload"]
        assert payload == {
            "trace_id": "trace-1",
            "value": True,
            "judge_name": "human-thumbs-up",
            "session_id": "session-1",
            "reason": "Looked correct.",
        }

    def test_submit_result_requires_exactly_one_judge_identifier(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError, match="Exactly one"):
            factory.submit_result(trace_id="trace-1", value=True)

    def test_submit_result_rejects_both_judge_identifiers(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError, match="Exactly one"):
            factory.submit_result(
                judge_id="judge-1",
                judge_name="human-thumbs-up",
                trace_id="trace-1",
                value=True,
            )

    def test_submit_result_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.submit_result(
            judge_id="judge-1", trace_id="trace-1", value=True
        )
        assert result is None

    def test_submit_result_validation_maps_to_validation_error(self):
        factory, client = _make_factory()
        client.post_projects_judge_results.side_effect = JudgmentAPIError(
            422, "value does not match judge score type", None
        )
        with pytest.raises(JudgmentValidationError):
            factory.submit_result(judge_id="judge-1", trace_id="trace-1", value=True)

    def test_submit_result_not_found_raises_raw_api_error(self):
        factory, client = _make_factory()
        client.post_projects_judge_results.side_effect = JudgmentAPIError(
            404, "trace not found", None
        )
        with pytest.raises(JudgmentAPIError):
            factory.submit_result(judge_id="judge-1", trace_id="trace-1", value=True)
