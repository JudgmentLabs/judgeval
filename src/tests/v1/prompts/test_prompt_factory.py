import pytest
from unittest.mock import MagicMock
from judgeval.v1.prompts.prompt_factory import PromptFactory
from judgeval.v1.prompts.prompt import Prompt


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def prompt_factory(mock_client):
    return PromptFactory(
        client=mock_client,
        project_id="test_project_id",
        project_name="test_project",
    )


class TestPromptFactoryCreate:
    def test_create_returns_prompt(self, prompt_factory, mock_client):
        mock_client.post_projects_prompts.return_value = {
            "created_at": "2024-01-01T00:00:00Z",
            "commit_id": "abc123",
            "parent_commit_id": None,
        }

        prompt = prompt_factory.create(name="test_prompt", prompt="Hello {{name}}")

        assert isinstance(prompt, Prompt)
        assert prompt.name == "test_prompt"
        assert prompt.prompt == "Hello {{name}}"
        assert prompt.commit_id == "abc123"
        mock_client.post_projects_prompts.assert_called_once()

    def test_create_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = PromptFactory(
            client=mock_client, project_id=None, project_name="test_project"
        )

        with caplog.at_level(logging.ERROR):
            prompt = factory.create(name="test_prompt", prompt="Hello")

        assert prompt is None
        assert "project_id is not set" in caplog.text
        assert "create()" in caplog.text
        mock_client.post_projects_prompts.assert_not_called()


class TestPromptFactoryGet:
    def test_get_returns_prompt_by_tag(self, prompt_factory, mock_client):
        mock_client.get_projects_prompts_by_name.return_value = {
            "commit": {
                "name": "test_prompt",
                "prompt": "Hello {{name}}",
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["production"],
                "commit_id": "abc123",
                "parent_commit_id": None,
                "first_name": "Test",
                "last_name": "User",
                "user_email": "test@example.com",
            }
        }

        prompt = prompt_factory.get(name="test_prompt", tag="production")

        assert isinstance(prompt, Prompt)
        assert prompt.name == "test_prompt"
        assert prompt.commit_id == "abc123"

    def test_get_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = PromptFactory(
            client=mock_client, project_id=None, project_name="test_project"
        )

        with caplog.at_level(logging.ERROR):
            prompt = factory.get(name="test_prompt", tag="production")

        assert prompt is None
        assert "project_id is not set" in caplog.text
        assert "get()" in caplog.text
        mock_client.get_projects_prompts_by_name.assert_not_called()


class TestPromptFactoryTag:
    def test_tag_returns_commit_id(self, prompt_factory, mock_client):
        mock_client.post_projects_prompts_by_name_tags.return_value = {
            "commit_id": "abc123"
        }

        result = prompt_factory.tag(
            name="test_prompt", commit_id="abc123", tags=["production"]
        )

        assert result == "abc123"
        mock_client.post_projects_prompts_by_name_tags.assert_called_once()

    def test_tag_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = PromptFactory(
            client=mock_client, project_id=None, project_name="test_project"
        )

        with caplog.at_level(logging.ERROR):
            result = factory.tag(
                name="test_prompt", commit_id="abc123", tags=["production"]
            )

        assert result is None
        assert "project_id is not set" in caplog.text
        assert "tag()" in caplog.text
        mock_client.post_projects_prompts_by_name_tags.assert_not_called()


class TestPromptFactoryUntag:
    def test_untag_returns_commit_ids(self, prompt_factory, mock_client):
        mock_client.delete_projects_prompts_by_name_tags.return_value = {
            "commit_ids": ["abc123", "def456"]
        }

        result = prompt_factory.untag(name="test_prompt", tags=["old_tag"])

        assert result == ["abc123", "def456"]
        mock_client.delete_projects_prompts_by_name_tags.assert_called_once()

    def test_untag_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = PromptFactory(
            client=mock_client, project_id=None, project_name="test_project"
        )

        with caplog.at_level(logging.ERROR):
            result = factory.untag(name="test_prompt", tags=["old_tag"])

        assert result is None
        assert "project_id is not set" in caplog.text
        assert "untag()" in caplog.text
        mock_client.delete_projects_prompts_by_name_tags.assert_not_called()


class TestPromptFactoryList:
    def test_list_returns_prompts(self, prompt_factory, mock_client):
        mock_client.get_projects_prompts_by_name_versions.return_value = {
            "versions": [
                {
                    "name": "test_prompt",
                    "prompt": "Hello v1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "tags": [],
                    "commit_id": "abc123",
                    "parent_commit_id": None,
                    "first_name": "Test",
                    "last_name": "User",
                    "user_email": "test@example.com",
                },
                {
                    "name": "test_prompt",
                    "prompt": "Hello v2",
                    "created_at": "2024-01-02T00:00:00Z",
                    "tags": ["latest"],
                    "commit_id": "def456",
                    "parent_commit_id": "abc123",
                    "first_name": "Test",
                    "last_name": "User",
                    "user_email": "test@example.com",
                },
            ]
        }

        prompts = prompt_factory.list(name="test_prompt")

        assert len(prompts) == 2
        assert all(isinstance(p, Prompt) for p in prompts)
        assert prompts[0].commit_id == "abc123"
        assert prompts[1].commit_id == "def456"

    def test_list_returns_none_when_project_id_missing(self, mock_client, caplog):
        import logging

        factory = PromptFactory(
            client=mock_client, project_id=None, project_name="test_project"
        )

        with caplog.at_level(logging.ERROR):
            prompts = factory.list(name="test_prompt")

        assert prompts is None
        assert "project_id is not set" in caplog.text
        assert "list()" in caplog.text
        mock_client.get_projects_prompts_by_name_versions.assert_not_called()
