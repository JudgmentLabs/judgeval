from __future__ import annotations

from unittest.mock import MagicMock


from judgeval.agent_judges.agent_judge import AgentJudge
from judgeval.agent_judges.agent_judge_factory import AgentJudgeFactory


def _make_factory(project_id="proj-1"):
    client = MagicMock()
    return AgentJudgeFactory(
        client=client, project_id=project_id, project_name="test"
    ), client


def _judge_detail(**overrides):
    base = {
        "id": "judge-1",
        "name": "helpfulness",
        "judge_description": "How helpful is the response?",
        "method": "LLM",
        "output": "",
        "last_updated": "2024-01-01",
        "score_type": "numeric",
        "behaviors": [],
        "model": "gpt-5.2",
        "prompt": "Rate helpfulness 0-1.",
        "description": None,
        "categories": None,
        "min_score": 0,
        "max_score": 1,
        "major_version": 0,
        "minor_version": 2,
        "versions": [],
    }
    base.update(overrides)
    return base


class TestAgentJudgeFactoryCreate:
    def test_create_returns_agent_judge(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {"judge_id": "judge-1"}
        result = factory.create(
            name="helpfulness",
            prompt="Rate helpfulness 0-1.",
            model="gpt-5.2",
            score_type="numeric",
        )
        assert isinstance(result, AgentJudge)
        assert result.judge_id == "judge-1"
        assert result.name == "helpfulness"
        assert result.prompt == "Rate helpfulness 0-1."
        assert result.model == "gpt-5.2"
        assert result.score_type == "numeric"

    def test_create_omits_unspecified_optional_fields(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {"judge_id": "judge-1"}
        factory.create(
            name="n",
            prompt="p",
            model="m",
            score_type="binary",
        )
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert set(payload.keys()) == {"name", "prompt", "model", "score_type"}

    def test_create_includes_optional_fields_when_provided(self):
        factory, client = _make_factory()
        client.post_projects_judges.return_value = {"judge_id": "judge-1"}
        factory.create(
            name="n",
            prompt="p",
            model="m",
            score_type="categorical",
            description="d",
            judge_description="jd",
            categories=[{"name": "Yes", "description": ""}],
            min_score=0.0,
            max_score=1.0,
        )
        payload = client.post_projects_judges.call_args.kwargs["payload"]
        assert payload["description"] == "d"
        assert payload["judge_description"] == "jd"
        assert payload["categories"] == [{"name": "Yes", "description": ""}]
        assert payload["min_score"] == 0.0
        assert payload["max_score"] == 1.0

    def test_create_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.create(
            name="n", prompt="p", model="m", score_type="numeric"
        )
        assert result is None


class TestAgentJudgeFactoryUpdate:
    def test_update_returns_agent_judge(self):
        factory, client = _make_factory()
        client.patch_projects_judges_by_judge_id.return_value = {
            "judge": _judge_detail(prompt="Updated prompt.", minor_version=3),
        }
        result = factory.update(
            judge_id="judge-1",
            prompt="Updated prompt.",
        )
        assert isinstance(result, AgentJudge)
        assert result.judge_id == "judge-1"
        assert result.prompt == "Updated prompt."
        assert result.minor_version == 3

    def test_update_omits_unset_fields(self):
        factory, client = _make_factory()
        client.patch_projects_judges_by_judge_id.return_value = {
            "judge": _judge_detail(),
        }
        factory.update(judge_id="judge-1", prompt="x")
        payload = client.patch_projects_judges_by_judge_id.call_args.kwargs[
            "payload"
        ]
        assert payload == {"prompt": "x"}

    def test_update_forwards_version_overrides(self):
        factory, client = _make_factory()
        client.patch_projects_judges_by_judge_id.return_value = {
            "judge": _judge_detail(),
        }
        factory.update(
            judge_id="judge-1",
            prompt="x",
            source_major_version=0,
            source_minor_version=1,
            target_major_version=1,
            target_minor_version=0,
        )
        payload = client.patch_projects_judges_by_judge_id.call_args.kwargs[
            "payload"
        ]
        assert payload["source_major_version"] == 0
        assert payload["source_minor_version"] == 1
        assert payload["target_major_version"] == 1
        assert payload["target_minor_version"] == 0

    def test_update_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.update(judge_id="judge-1", prompt="x")
        assert result is None
