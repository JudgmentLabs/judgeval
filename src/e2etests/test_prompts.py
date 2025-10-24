"""
Tests for prompt operations in the JudgmentClient.
"""

import uuid
import pytest
from judgeval import JudgmentClient
from judgeval.prompt import Prompt
from judgeval.exceptions import JudgmentAPIError


def test_create_prompt(client: JudgmentClient, project_name: str, random_name: str):
    """Test prompt creation with tags."""
    prompt = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content",
        tags=["test", "test2"],
    )
    assert prompt is not None, "Failed to create prompt"
    assert prompt.name == random_name, "Prompt name should match"
    assert prompt.commit_id is not None, "Prompt should have a commit_id"


def test_create_prompt_new_version(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test creating a new version of an existing prompt."""
    prompt_v1 = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt version 1",
        tags=["v1", "test"],
    )
    assert prompt_v1 is not None, "Failed to create first prompt version"
    commit_id_v1 = prompt_v1.commit_id

    prompt_v2 = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt version 2",
        tags=["v2", "test"],
    )
    assert prompt_v2 is not None, "Failed to create second prompt version"
    assert prompt_v2.commit_id != commit_id_v1, (
        "New version should have different commit_id"
    )


def test_get_prompt_by_tag(client: JudgmentClient, project_name: str, random_name: str):
    """Test retrieving a prompt by tag."""
    Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content",
        tags=["specific-tag"],
    )

    retrieved_prompt = Prompt.get(
        project_name=project_name,
        name=random_name,
        tag="specific-tag",
    )
    assert retrieved_prompt is not None, "Failed to retrieve prompt by tag"
    assert retrieved_prompt.name == random_name, (
        "Retrieved prompt should have correct name"
    )


def test_get_prompt_by_commit_id(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test retrieving a prompt by commit_id."""
    prompt = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content",
        tags=["specific-tag"],
    )

    retrieved_prompt = Prompt.get(
        project_name=project_name,
        name=random_name,
        commit_id=prompt.commit_id,
    )
    assert retrieved_prompt is not None, "Failed to retrieve prompt by commit_id"
    assert retrieved_prompt.name == random_name, (
        "Retrieved prompt should have correct name"
    )
    assert retrieved_prompt.commit_id == prompt.commit_id, (
        "Retrieved prompt should have correct commit_id"
    )
    assert retrieved_prompt.prompt == "test prompt content", (
        "Retrieved prompt should have correct prompt"
    )


def test_get_prompt_latest(client: JudgmentClient, project_name: str, random_name: str):
    """Test retrieving the latest version of a prompt without specifying tag or commit_id."""
    Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="version 1",
        tags=["v1"],
    )

    Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="version 2",
        tags=["v2"],
    )

    latest_prompt = Prompt.get(
        project_name=project_name,
        name=random_name,
    )
    assert latest_prompt is not None, "Failed to retrieve latest prompt"
    assert latest_prompt.prompt == "version 2", "Should retrieve the latest version"


def test_tag_prompt_commit(client: JudgmentClient, project_name: str, random_name: str):
    """Test adding tags to an existing prompt commit."""
    prompt = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content",
        tags=["original-tag"],
    )
    commit_id = prompt.commit_id

    new_tag = f"dynamic-tag-{uuid.uuid4().hex[:8]}"
    tagged_prompt = Prompt.tag(
        project_name=project_name,
        name=random_name,
        commit_id=commit_id,
        tags=[new_tag],
    )
    assert tagged_prompt is not None, "Failed to tag prompt"

    retrieved_by_new_tag = Prompt.get(
        project_name=project_name,
        name=random_name,
        tag=new_tag,
    )
    assert retrieved_by_new_tag is not None, (
        "Should be able to retrieve prompt by new tag"
    )
    assert retrieved_by_new_tag.commit_id == commit_id, (
        "Retrieved prompt should have same commit_id"
    )

    retrieved_by_original_tag = Prompt.get(
        project_name=project_name,
        name=random_name,
        tag="original-tag",
    )
    assert retrieved_by_original_tag is not None, "Original tag should still work"
    assert retrieved_by_original_tag.commit_id == commit_id, (
        "Should retrieve same commit"
    )


def test_untag_prompt(client: JudgmentClient, project_name: str, random_name: str):
    """Test untagging a prompt."""
    prompt = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content",
        tags=["tag-1", "tag-2"],
    )
    prompt2 = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content 2",
        tags=["tag-3", "tag-4"],
    )
    untagged_prompt = Prompt.untag(
        project_name=project_name,
        name=random_name,
        tags=["tag-1"],
    )
    new_prompt_single = Prompt.get(
        project_name=project_name,
        name=random_name,
        commit_id=prompt.commit_id,
    )
    assert untagged_prompt is not None, "Failed to untag prompt"
    assert len(untagged_prompt) == 1, "Should untag one prompt"
    assert untagged_prompt[0] == prompt.commit_id, "Should untag the correct commit"
    assert "tag-1" not in new_prompt_single.tags, "Original tag should be removed"

    untagged_prompt2 = Prompt.untag(
        project_name=project_name,
        name=random_name,
        tags=["tag-2", "tag-3", "tag-4"],
    )

    new_prompt_double = Prompt.get(
        project_name=project_name,
        name=random_name,
        commit_id=prompt.commit_id,
    )
    new_prompt_double_2 = Prompt.get(
        project_name=project_name,
        name=random_name,
        commit_id=prompt2.commit_id,
    )
    assert untagged_prompt2 is not None, "Failed to untag prompt"
    assert len(untagged_prompt2) == 2, "Should untag two prompts"
    assert set(untagged_prompt2) == {prompt.commit_id, prompt2.commit_id}, (
        "Should untag the correct commits"
    )
    assert "tag-2" not in new_prompt_double.tags, "Original tag should be removed"
    assert "tag-3" not in new_prompt_double_2.tags, "Original tag should be removed"
    assert "tag-4" not in new_prompt_double_2.tags, "Original tag should be removed"


def test_overwrite_tag(client: JudgmentClient, project_name: str, random_name: str):
    """Test overwriting a tag."""
    prompt = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content",
        tags=["tag-1"],
    )
    prompt2 = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content 2",
    )
    tagged_prompt = Prompt.tag(
        project_name=project_name,
        name=random_name,
        commit_id=prompt2.commit_id,
        tags=["tag-1"],
    )
    new_prompt1 = Prompt.get(
        project_name=project_name,
        name=random_name,
        commit_id=prompt.commit_id,
    )
    new_prompt2 = Prompt.get(
        project_name=project_name,
        name=random_name,
        commit_id=prompt2.commit_id,
    )
    assert tagged_prompt and tagged_prompt == prompt2.commit_id, (
        "Should tag second commit"
    )
    assert "tag-1" in new_prompt2.tags, "tag-1 should have been added to second commit"
    assert "tag-1" not in new_prompt1.tags, (
        "tag-1 should have been removed from first commit"
    )


def test_list_prompts(client: JudgmentClient, project_name: str, random_name: str):
    """Test listing all versions of a prompt."""
    prompt_name = random_name

    Prompt.create(
        project_name=project_name,
        name=prompt_name,
        prompt="version 1",
        tags=["v1"],
    )

    Prompt.create(
        project_name=project_name,
        name=prompt_name,
        prompt="version 2",
        tags=["v2"],
    )

    Prompt.create(
        project_name=project_name,
        name=prompt_name,
        prompt="version 3",
        tags=["v3"],
    )

    prompt_list = Prompt.list(
        project_name=project_name,
        name=prompt_name,
    )
    assert prompt_list is not None, "Failed to list prompts"
    assert len(prompt_list) == 3, f"Expected 3 versions, got {len(prompt_list)}"
    assert prompt_list[0].prompt == "version 3", "Third prompt should be version 3"
    assert prompt_list[1].prompt == "version 2", "Second prompt should be version 2"
    assert prompt_list[2].prompt == "version 1", "First prompt should be version 1"


def test_multiple_tags_on_creation(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test creating a prompt with multiple tags."""
    tags = ["tag1", "tag2", "tag3"]
    prompt = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt with multiple tags",
        tags=tags,
    )
    commit_id = prompt.commit_id

    for tag in tags:
        retrieved = Prompt.get(
            project_name=project_name,
            name=random_name,
            tag=tag,
        )
        assert retrieved is not None, (
            f"Should be able to retrieve prompt by tag '{tag}'"
        )
        assert retrieved.commit_id == commit_id, (
            f"Retrieved prompt by tag '{tag}' should have correct commit_id"
        )


def test_compile_prompt(client: JudgmentClient, project_name: str, random_name: str):
    """Test compiling a prompt."""
    prompt = Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt {{name}}",
        tags=["tag-1"],
    )
    compiled_prompt = prompt.compile(name=random_name)
    assert compiled_prompt == f"test prompt {random_name}", (
        "Compiled prompt should contain the random_name"
    )


def test_get_nonexistent_prompt(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test getting a nonexistent prompt."""
    with pytest.raises(JudgmentAPIError) as exc_info:
        Prompt.get(
            project_name=project_name,
            name=random_name,
            tag="nonexistent_tag",
        )
    assert exc_info.value.status_code == 404, (
        "Should raise 404 error for nonexistent prompt"
    )
    with pytest.raises(JudgmentAPIError) as exc_info:
        Prompt.get(
            project_name=project_name,
            name=random_name,
            commit_id="nonexistent_commit_id",
        )
    assert exc_info.value.status_code == 404, (
        "Should raise 404 error for nonexistent prompt"
    )
    with pytest.raises(JudgmentAPIError) as exc_info:
        Prompt.get(
            project_name=project_name,
            name=random_name,
        )
    assert exc_info.value.status_code == 404, (
        "Should raise 404 error for nonexistent prompt"
    )


def test_remove_nonexistent_tag(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test removing a nonexistent tag."""
    Prompt.create(
        project_name=project_name,
        name=random_name,
        prompt="test prompt content",
        tags=["tag-1"],
    )
    with pytest.raises(JudgmentAPIError) as exc_info:
        Prompt.untag(
            project_name=project_name,
            name=random_name,
            tags=["nonexistent_tag"],
        )
    assert exc_info.value.status_code == 404, (
        "Should raise 404 error for nonexistent tag"
    )


def test_tag_nonexistent_prompt(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test tagging a nonexistent prompt."""
    with pytest.raises(JudgmentAPIError) as exc_info:
        Prompt.tag(
            project_name=project_name,
            name=random_name,
            commit_id="nonexistent_commit_id",
            tags=["tag-1"],
        )
    assert exc_info.value.status_code == 404, (
        "Should raise 404 error for nonexistent prompt"
    )


def test_untag_nonexistent_prompt(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test untagging a nonexistent prompt."""
    with pytest.raises(JudgmentAPIError) as exc_info:
        Prompt.untag(
            project_name=project_name,
            name=random_name,
            tags=["tag-1"],
        )
    assert exc_info.value.status_code == 404, (
        "Should raise 404 error for nonexistent prompt"
    )


def test_tag_with_no_tags(client: JudgmentClient, project_name: str, random_name: str):
    """Test tagging a prompt with no tags."""
    with pytest.raises(JudgmentAPIError) as exc_info:
        Prompt.tag(
            project_name=project_name,
            name=random_name,
            commit_id="nonexistent_commit_id",
            tags=[],
        )
    assert exc_info.value.status_code == 422, "Should raise 422 error for no tags"


def test_untag_with_no_tags(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test untagging a prompt with no tags."""
    with pytest.raises(JudgmentAPIError) as exc_info:
        Prompt.untag(
            project_name=project_name,
            name=random_name,
            tags=[],
        )
    assert exc_info.value.status_code == 422, "Should raise 422 error for no tags"
