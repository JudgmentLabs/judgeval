import pytest
from unittest.mock import MagicMock
from judgeval.v1.scorers.prompt_scorer.prompt_scorer_factory import PromptScorerFactory
from judgeval.v1.scorers.prompt_scorer.prompt_scorer import PromptScorer


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.organization_id = "test_org"
    client.api_key = "test_key"
    return client


class TestPromptScorerFactoryGet:
    def test_get_returns_scorer_when_exists(self, mock_client):
        mock_client.get_projects_scorers.return_value = {
            "scorers": [
                {
                    "prompt": "Test prompt",
                    "threshold": 0.7,
                    "options": None,
                    "model": "gpt-4",
                    "description": "Test description",
                    "is_trace": False,
                }
            ]
        }

        factory = PromptScorerFactory(
            client=mock_client, is_trace=False, project_id="test_project_id"
        )
        PromptScorerFactory._cache.clear()
        scorer = factory.get("TestScorer")

        assert isinstance(scorer, PromptScorer)
        assert scorer._name == "TestScorer"
        assert scorer._prompt == "Test prompt"
        assert scorer._threshold == 0.7

    def test_get_returns_none_when_not_found(self, mock_client):
        mock_client.get_projects_scorers.return_value = {"scorers": []}

        factory = PromptScorerFactory(
            client=mock_client, is_trace=False, project_id="test_project_id"
        )
        PromptScorerFactory._cache.clear()
        scorer = factory.get("NonExistentScorer")

        assert scorer is None

    def test_get_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = PromptScorerFactory(
            client=mock_client, is_trace=False, project_id=None
        )

        with caplog.at_level(logging.ERROR):
            scorer = factory.get("TestScorer")

        assert scorer is None
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
                    "is_trace": False,
                }
            ]
        }

        factory = PromptScorerFactory(
            client=mock_client, is_trace=False, project_id="test_project_id"
        )
        PromptScorerFactory._cache.clear()

        scorer1 = factory.get("TestScorer")
        scorer2 = factory.get("TestScorer")

        assert scorer1 is not None
        assert scorer2 is not None
        assert mock_client.get_projects_scorers.call_count == 1
