import pytest
from unittest.mock import MagicMock
from judgeval.v1.scorers.custom_scorer.custom_scorer_factory import CustomScorerFactory
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer
from judgeval.exceptions import JudgmentAPIError


@pytest.fixture
def mock_client():
    return MagicMock()


class TestCustomScorerFactoryGet:
    def test_get_returns_scorer_when_exists(self, mock_client):
        mock_client.get_projects_scorers_custom_by_name_exists.return_value = {
            "exists": True
        }

        factory = CustomScorerFactory(client=mock_client, project_id="test_project_id")
        scorer = factory.get("TestScorer")

        assert isinstance(scorer, CustomScorer)
        assert scorer._name == "TestScorer"
        assert scorer._project_id == "test_project_id"

    def test_get_raises_when_scorer_not_exists(self, mock_client):
        mock_client.get_projects_scorers_custom_by_name_exists.return_value = {
            "exists": False
        }

        factory = CustomScorerFactory(client=mock_client, project_id="test_project_id")

        with pytest.raises(JudgmentAPIError) as exc_info:
            factory.get("NonExistentScorer")

        assert exc_info.value.status_code == 404
        assert "NonExistentScorer" in str(exc_info.value)

    def test_get_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = CustomScorerFactory(client=mock_client, project_id=None)

        with caplog.at_level(logging.ERROR):
            scorer = factory.get("TestScorer")

        assert scorer is None
        assert "project_id is not set" in caplog.text
        assert "get()" in caplog.text
        mock_client.get_projects_scorers_custom_by_name_exists.assert_not_called()

    def test_get_propagates_api_error(self, mock_client):
        mock_client.get_projects_scorers_custom_by_name_exists.side_effect = Exception(
            "API Error"
        )

        factory = CustomScorerFactory(client=mock_client, project_id="test_project_id")

        with pytest.raises(Exception, match="API Error"):
            factory.get("TestScorer")


class TestCustomScorerFactoryUpload:
    def test_upload_returns_false_when_project_id_missing(
        self, mock_client, caplog, tmp_path
    ):
        import logging

        scorer_file = tmp_path / "scorer.py"
        scorer_file.write_text(
            """
from judgeval.v1.scorers.base_scorer import ExampleScorer

class TestScorer(ExampleScorer):
    pass
"""
        )

        factory = CustomScorerFactory(client=mock_client, project_id=None)

        with caplog.at_level(logging.ERROR):
            result = factory.upload(str(scorer_file))

        assert result is False
        assert "project_id is not set" in caplog.text
        assert "upload()" in caplog.text
        mock_client.post_projects_scorers_custom.assert_not_called()
