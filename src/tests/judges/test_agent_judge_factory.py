from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from judgeval.agent_judges.agent_judge import AgentJudge
from judgeval.agent_judges.agent_judge_factory import AgentJudgeFactory

ADVANCED_SETTINGS = {
    "online_environment_names": None,
    "online_evaluation_mode": "on_demand",
    "online_sampling_rate": 100,
    "online_span_triggers": None,
    "online_session_scoring": False,
    "session_evaluation_count_enabled": None,
    "session_evaluation_idle_enabled": None,
    "session_evaluation_min_candidate_traces": None,
    "session_evaluation_max_wait_minutes": None,
}

DEFINITION = {
    "id": "def-1",
    "description": None,
    "prompt": "Rate helpfulness 0-1.",
    "reasoning_level": "lite",
    "min_score": 0,
    "max_score": 1,
    "model": "gpt-5.2",
    "threshold": 0.5,
}


def _judge_detail(**overrides):
    return {
        "id": "judge-1",
        "name": "helpfulness",
        "judge_description": "How helpful is the response?",
        "judge_type": "prompt",
        "score_type": "numeric",
        "outputs": [],
        "judge_major_version": 1,
        "judge_minor_version": 2,
        **ADVANCED_SETTINGS,
        "definition": DEFINITION,
        **overrides,
    }


PASS_FAIL = {
    "scoreType": "categorical",
    "outputs": [
        {"name": "Pass", "description": "The output meets the rubric."},
        {"name": "Fail", "description": "The output does not meet the rubric."},
    ],
}


CATEGORICAL = _judge_detail(
    score_type="categorical",
    outputs=[
        {"behavior_id": "b-yes", "name": "Yes", "description": "y"},
        {"behavior_id": "b-no", "name": "No", "description": "n"},
    ],
)


def _make_factory(project_id="proj-1", lookups=None):
    client = MagicMock()
    client.post_projects_judges.return_value = {"judge_id": "judge-1"}
    responses = [{"judge": j} for j in (lookups or [_judge_detail()])]
    client.post_projects_judges_by_judge_id_lookup.side_effect = (
        responses + [{"judge": _judge_detail()}] * 3
    )
    factory = AgentJudgeFactory(
        client=client, project_id=project_id, project_name="test"
    )
    return factory, client


def _patch_body(client):
    return client.patch_projects_judges_by_judge_id.call_args.kwargs["payload"]


def _lookup_versions(client):
    return [
        call.kwargs["payload"]["version"]
        for call in client.post_projects_judges_by_judge_id_lookup.call_args_list
    ]


class TestAgentJudgeFactoryCreate:
    def test_create_sends_prompt_judge_body(self):
        factory, client = _make_factory()
        factory.create(
            name="n",
            prompt="p",
            score_type="binary",
            reasoning_level="high",
            description="d",
            judge_description="jd",
            min_score=1,
            max_score=5,
        )
        assert client.post_projects_judges.call_args.kwargs == {
            "project_id": "proj-1",
            "payload": {
                "judge": {
                    "name": "n",
                    "judgeDescription": "jd",
                    "judgeType": "prompt",
                    "initialVersion": PASS_FAIL,
                    "config": {
                        "prompt": "p",
                        "description": "d",
                        "reasoningLevel": "high",
                        "minScore": 1,
                        "maxScore": 5,
                    },
                }
            },
        }

    def test_create_omits_unspecified_optional_fields(self):
        factory, client = _make_factory()
        factory.create(name="n", prompt="p", score_type="binary")
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload == {
            "judge": {
                "name": "n",
                "judgeType": "prompt",
                "initialVersion": PASS_FAIL,
                "config": {"prompt": "p"},
            }
        }

    def test_create_sends_categories_as_outputs(self):
        factory, client = _make_factory()
        categories = [
            {"name": "Yes", "description": "y"},
            {"name": "No", "description": "n"},
        ]
        factory.create(
            name="n", prompt="p", score_type="categorical", categories=categories
        )
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload["judge"]["initialVersion"] == {
            "scoreType": "categorical",
            "outputs": categories,
        }

    def test_create_rejects_numeric_before_calling_the_api(self):
        factory, client = _make_factory()
        with pytest.raises(ValueError, match="must be categorical"):
            factory.create(name="n", prompt="p", score_type="numeric")
        client.post_projects_judges.assert_not_called()

    def test_create_returns_judge_as_looked_up(self):
        factory, _ = _make_factory()
        result = factory.create(
            name="helpfulness",
            prompt="Rate helpfulness 0-1.",
            model="ignored",
            score_type="binary",
        )
        assert result == AgentJudge(
            judge_id="judge-1",
            name="helpfulness",
            prompt="Rate helpfulness 0-1.",
            model="gpt-5.2",
            reasoning_level="lite",
            score_type="numeric",
            description=None,
            judge_description="How helpful is the response?",
            categories=None,
            min_score=0,
            max_score=1,
            major_version=1,
            minor_version=2,
        )

    def test_create_returns_none_without_calling_api_when_project_unresolved(self):
        factory, client = _make_factory(project_id=None)
        assert factory.create(name="n", prompt="p", score_type="binary") is None
        client.post_projects_judges.assert_not_called()


class TestAgentJudgeFactoryUpdate:
    def test_update_writes_complete_version_bumped_from_latest(self):
        factory, client = _make_factory()
        factory.update(judge_id="judge-1", prompt="Updated.")
        assert _patch_body(client) == {
            "update": {
                "newVersion": {
                    "targetVersion": {"major": 1, "minor": 3},
                    "promoteToProduction": True,
                    "judgeType": "prompt",
                    "scoreType": "numeric",
                    "config": {
                        "prompt": "Updated.",
                        "description": None,
                        "reasoningLevel": "lite",
                        "minScore": 0,
                        "maxScore": 1,
                    },
                    "advancedSettings": ADVANCED_SETTINGS,
                }
            }
        }

    def test_update_judge_description_only_writes_no_version(self):
        factory, client = _make_factory()
        factory.update(judge_id="judge-1", judge_description="new")
        assert _patch_body(client) == {"update": {"judgeDescription": "new"}}

    def test_update_copies_source_and_writes_target(self):
        factory, client = _make_factory(
            lookups=[
                _judge_detail(),
                _judge_detail(definition={**DEFINITION, "prompt": "v0.1"}),
            ]
        )
        factory.update(
            judge_id="judge-1",
            source_major_version=0,
            source_minor_version=1,
            target_major_version=2,
            target_minor_version=0,
        )
        new_version = _patch_body(client)["update"]["newVersion"]
        assert (new_version["targetVersion"], new_version["config"]["prompt"]) == (
            {"major": 2, "minor": 0},
            "v0.1",
        )

    def test_update_source_minor_alone_reads_from_latest_major(self):
        factory, client = _make_factory()
        factory.update(judge_id="judge-1", source_minor_version=1)
        assert _lookup_versions(client)[1] == {"major": 1, "minor": 1}

    def test_update_target_major_alone_bumps_latest_minor(self):
        factory, client = _make_factory()
        factory.update(judge_id="judge-1", prompt="x", target_major_version=1)
        assert _patch_body(client)["update"]["newVersion"]["targetVersion"] == {
            "major": 1,
            "minor": 3,
        }

    def test_update_requires_source_minor_for_an_older_major(self):
        factory, client = _make_factory()
        with pytest.raises(ValueError, match="source_minor_version is required"):
            factory.update(judge_id="judge-1", source_major_version=0)
        client.patch_projects_judges_by_judge_id.assert_not_called()

    def test_update_binary_on_a_pass_fail_judge_stays_categorical(self):
        factory, client = _make_factory(lookups=[CATEGORICAL])
        factory.update(judge_id="judge-1", prompt="x", score_type="binary")
        assert _patch_body(client)["update"]["newVersion"]["scoreType"] == (
            "categorical"
        )

    def test_update_reuses_behaviors_for_matching_category_names(self):
        factory, client = _make_factory(lookups=[CATEGORICAL])
        factory.update(
            judge_id="judge-1",
            categories=[
                {"name": "Yes", "description": "y"},
                {"name": "Maybe", "description": "m"},
            ],
        )
        assert _patch_body(client)["update"]["newVersion"]["outputs"] == [
            {"behaviorId": "b-yes"},
            {"name": "Maybe", "description": "m"},
        ]

    def test_update_rejects_a_changed_category_description(self):
        factory, client = _make_factory(lookups=[CATEGORICAL])
        with pytest.raises(ValueError, match="different description"):
            factory.update(
                judge_id="judge-1",
                categories=[{"name": "Yes", "description": "changed"}],
            )
        client.patch_projects_judges_by_judge_id.assert_not_called()

    def test_update_copies_action_config_from_source(self):
        groups = {"action_groups": []}
        factory, client = _make_factory(
            lookups=[_judge_detail(), _judge_detail(action_config=groups)]
        )
        factory.update(
            judge_id="judge-1", source_major_version=1, source_minor_version=0
        )
        assert _patch_body(client)["update"]["newVersion"]["actionConfig"] == groups

    def test_update_keeps_existing_outputs_without_categories(self):
        factory, client = _make_factory(lookups=[CATEGORICAL])
        factory.update(judge_id="judge-1", prompt="x")
        assert _patch_body(client)["update"]["newVersion"]["outputs"] == [
            {"behaviorId": "b-yes"},
            {"behaviorId": "b-no"},
        ]

    def test_update_returns_written_version(self):
        factory, _ = _make_factory(
            lookups=[_judge_detail(), _judge_detail(judge_minor_version=3)]
        )
        result = factory.update(judge_id="judge-1", prompt="x")
        assert result.minor_version == 3

    def test_update_rejects_non_prompt_judges(self):
        factory, _ = _make_factory(lookups=[_judge_detail(judge_type="custom")])
        with pytest.raises(ValueError, match="only prompt-based agent judges"):
            factory.update(judge_id="judge-1", prompt="x")

    def test_update_returns_none_without_calling_api_when_project_unresolved(self):
        factory, client = _make_factory(project_id=None)
        assert factory.update(judge_id="judge-1", prompt="x") is None
        client.patch_projects_judges_by_judge_id.assert_not_called()
