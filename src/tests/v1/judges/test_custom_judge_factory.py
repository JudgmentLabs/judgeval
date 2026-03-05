import pytest
from unittest.mock import MagicMock
from judgeval.v1.judges.custom_judge_factory import CustomJudgeFactory
from judgeval.v1.judges.custom_judge import CustomJudge
from judgeval.exceptions import JudgmentAPIError


@pytest.fixture
def mock_client():
    return MagicMock()


class TestCustomJudgeFactoryGet:
    def test_get_returns_judge_when_exists(self, mock_client):
        mock_client.get_projects_scorers_custom_by_name_exists.return_value = {
            "exists": True
        }

        factory = CustomJudgeFactory(client=mock_client, project_id="test_project_id")
        judge = factory.get("TestJudge")

        assert isinstance(judge, CustomJudge)
        assert judge._name == "TestJudge"
        assert judge._project_id == "test_project_id"

    def test_get_raises_when_judge_not_exists(self, mock_client):
        mock_client.get_projects_scorers_custom_by_name_exists.return_value = {
            "exists": False
        }

        factory = CustomJudgeFactory(client=mock_client, project_id="test_project_id")

        with pytest.raises(JudgmentAPIError) as exc_info:
            factory.get("NonExistentJudge")

        assert exc_info.value.status_code == 404
        assert "NonExistentJudge" in str(exc_info.value)

    def test_get_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = CustomJudgeFactory(client=mock_client, project_id=None)

        with caplog.at_level(logging.ERROR):
            judge = factory.get("TestJudge")

        assert judge is None
        assert "project_id is not set" in caplog.text
        assert "get()" in caplog.text
        mock_client.get_projects_scorers_custom_by_name_exists.assert_not_called()

    def test_get_propagates_api_error(self, mock_client):
        mock_client.get_projects_scorers_custom_by_name_exists.side_effect = Exception(
            "API Error"
        )

        factory = CustomJudgeFactory(client=mock_client, project_id="test_project_id")

        with pytest.raises(Exception, match="API Error"):
            factory.get("TestJudge")


class TestCustomJudgeFactoryUpload:
    def test_upload_returns_false_when_project_id_missing(
        self, mock_client, caplog, tmp_path
    ):
        import logging

        scorer_file = tmp_path / "scorer.py"
        scorer_file.write_text(
            """
from judgeval.v1.judges import Judge

class TestJudge(Judge):
    pass
"""
        )

        factory = CustomJudgeFactory(client=mock_client, project_id=None)

        with caplog.at_level(logging.ERROR):
            result = factory.upload(str(scorer_file))

        assert result is False
        assert "project_id is not set" in caplog.text
        assert "upload()" in caplog.text
        mock_client.post_projects_scorers_custom.assert_not_called()
