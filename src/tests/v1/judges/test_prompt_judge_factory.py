import pytest
from unittest.mock import MagicMock
from judgeval.v1.judges.prompt_judge_factory import PromptJudgeFactory
from judgeval.v1.judges.prompt_judge import PromptJudge


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.organization_id = "test_org"
    client.api_key = "test_key"
    return client


class TestPromptJudgeFactoryGet:
    def test_get_returns_judge_when_exists(self, mock_client):
        mock_client.get_projects_scorers.return_value = {
            "scorers": [
                {
                    "prompt": "Test prompt",
                    "threshold": 0.7,
                    "options": None,
                    "model": "gpt-4",
                    "description": "Test description",
                }
            ]
        }

        factory = PromptJudgeFactory(client=mock_client, project_id="test_project_id")
        PromptJudgeFactory._cache.clear()
        judge = factory.get("TestJudge")

        assert isinstance(judge, PromptJudge)
        assert judge._name == "TestJudge"
        assert judge._prompt == "Test prompt"
        assert judge._threshold == 0.7

    def test_get_returns_none_when_not_found(self, mock_client):
        mock_client.get_projects_scorers.return_value = {"scorers": []}

        factory = PromptJudgeFactory(client=mock_client, project_id="test_project_id")
        PromptJudgeFactory._cache.clear()
        judge = factory.get("NonExistentJudge")

        assert judge is None

    def test_get_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = PromptJudgeFactory(client=mock_client, project_id=None)

        with caplog.at_level(logging.ERROR):
            judge = factory.get("TestJudge")

        assert judge is None
        assert "project_id is not set" in caplog.text
        assert "get()" in caplog.text
        mock_client.get_projects_scorers.assert_not_called()

    def test_get_caches_results(self, mock_client):
        mock_client.get_projects_scorers.return_value = {
            "scorers": [
                {
                    "prompt": "Test prompt",
                    "threshold": 0.5,
                    "options": None,
                    "model": None,
                    "description": None,
                }
            ]
        }

        factory = PromptJudgeFactory(client=mock_client, project_id="test_project_id")
        PromptJudgeFactory._cache.clear()

        judge1 = factory.get("TestJudge")
        judge2 = factory.get("TestJudge")

        assert judge1 is not None
        assert judge2 is not None
        assert mock_client.get_projects_scorers.call_count == 1
