from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from judgeval.v1.prompts.prompt import Prompt
from judgeval.v1.prompts.prompt_factory import PromptFactory


def _make_factory(project_id="proj-1"):
    client = MagicMock()
    return PromptFactory(
        client=client, project_id=project_id, project_name="test"
    ), client


class TestPromptFactoryCreate:
    def test_create_returns_prompt(self):
        factory, client = _make_factory()
        client.post_projects_prompts.return_value = {
            "created_at": "2024-01-01",
            "commit_id": "abc123",
        }
        p = factory.create("my-prompt", "Hello {{name}}")
        assert isinstance(p, Prompt)
        assert p.name == "my-prompt"
        assert p.prompt == "Hello {{name}}"
        assert p.commit_id == "abc123"

    def test_create_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.create("p", "text")
        assert result is None

    def test_create_passes_tags(self):
        factory, client = _make_factory()
        client.post_projects_prompts.return_value = {
            "created_at": "2024-01-01",
            "commit_id": "c1",
        }
        factory.create("p", "text", tags=["v1", "production"])
        _, kwargs = client.post_projects_prompts.call_args
        assert "v1" in kwargs["payload"]["tags"]

    def test_create_raises_on_client_error(self):
        factory, client = _make_factory()
        client.post_projects_prompts.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError):
            factory.create("p", "text")


class TestPromptFactoryGet:
    def test_get_by_commit_id_returns_prompt(self):
        factory, client = _make_factory()
        client.get_projects_prompts_by_name.return_value = {
            "commit": {
                "name": "my-prompt",
                "prompt": "Hello",
                "created_at": "2024",
                "tags": [],
                "commit_id": "c1",
                "first_name": "John",
                "last_name": "Doe",
                "user_email": "john@example.com",
            }
        }
        p = factory.get(name="my-prompt", commit_id="c1")
        assert isinstance(p, Prompt)
        assert p.commit_id == "c1"

    def test_get_missing_commit_returns_none(self):
        factory, client = _make_factory()
        client.get_projects_prompts_by_name.return_value = {"commit": None}
        result = factory.get(name="missing")
        assert result is None

    def test_get_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.get(name="p")
        assert result is None

    def test_get_both_commit_id_and_tag_returns_none(self):
        factory, _ = _make_factory()
        result = factory.get(name="p", commit_id="c1", tag="v1")
        assert result is None


class TestPromptFactoryTag:
    def test_tag_returns_commit_id(self):
        factory, client = _make_factory()
        client.post_projects_prompts_by_name_tags.return_value = {"commit_id": "c2"}
        result = factory.tag("p", "c1", ["production"])
        assert result == "c2"

    def test_tag_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.tag("p", "c1", ["v1"])
        assert result is None


class TestPromptFactoryList:
    def test_list_returns_prompts(self):
        factory, client = _make_factory()
        client.get_projects_prompts_by_name_versions.return_value = {
            "versions": [
                {
                    "name": "p",
                    "prompt": "text",
                    "tags": [],
                    "created_at": "2024",
                    "commit_id": "c1",
                    "first_name": "J",
                    "last_name": "D",
                    "user_email": "j@d.com",
                }
            ]
        }
        result = factory.list("p")
        assert len(result) == 1
        assert isinstance(result[0], Prompt)

    def test_list_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.list("p")
        assert result is None


class TestPrompt:
    def test_compile_substitutes_variables(self):
        p = Prompt(
            name="greet",
            prompt="Hello {{name}}!",
            created_at="2024",
            tags=[],
            commit_id="c1",
        )
        assert p.compile(name="World") == "Hello World!"

    def test_compile_missing_variable_raises(self):
        p = Prompt(
            name="greet",
            prompt="Hello {{name}}!",
            created_at="2024",
            tags=[],
            commit_id="c1",
        )
        with pytest.raises(ValueError, match="name"):
            p.compile()

    def test_compile_no_variables(self):
        p = Prompt(
            name="simple",
            prompt="Static text",
            created_at="2024",
            tags=[],
            commit_id="c1",
        )
        assert p.compile() == "Static text"
