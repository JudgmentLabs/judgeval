import uuid
import pytest
from judgeval.v1 import Judgeval
from judgeval.exceptions import JudgmentAPIError


def test_create_prompt(client: Judgeval, random_name: str):
    prompt = client.prompts.create(
        name=random_name,
        prompt="test prompt content",
        tags=["test", "test2"],
    )
    assert prompt is not None, "Failed to create prompt"
    assert prompt.name == random_name
    assert prompt.commit_id is not None


def test_create_prompt_new_version(client: Judgeval, random_name: str):
    prompt_v1 = client.prompts.create(
        name=random_name,
        prompt="test prompt version 1",
        tags=["v1", "test"],
    )
    assert prompt_v1 is not None
    commit_id_v1 = prompt_v1.commit_id

    prompt_v2 = client.prompts.create(
        name=random_name,
        prompt="test prompt version 2",
        tags=["v2", "test"],
    )
    assert prompt_v2 is not None
    assert prompt_v2.commit_id != commit_id_v1


def test_get_prompt_by_tag(client: Judgeval, random_name: str):
    client.prompts.create(
        name=random_name,
        prompt="test prompt content",
        tags=["specific-tag"],
    )

    retrieved_prompt = client.prompts.get(
        name=random_name,
        tag="specific-tag",
    )
    assert retrieved_prompt is not None
    assert retrieved_prompt.name == random_name


def test_get_prompt_by_commit_id(client: Judgeval, random_name: str):
    prompt = client.prompts.create(
        name=random_name,
        prompt="test prompt content",
        tags=["specific-tag"],
    )

    retrieved_prompt = client.prompts.get(
        name=random_name,
        commit_id=prompt.commit_id,
    )
    assert retrieved_prompt is not None
    assert retrieved_prompt.name == random_name
    assert retrieved_prompt.commit_id == prompt.commit_id
    assert retrieved_prompt.prompt == "test prompt content"


def test_get_prompt_latest(client: Judgeval, random_name: str):
    client.prompts.create(
        name=random_name,
        prompt="version 1",
        tags=["v1"],
    )

    client.prompts.create(
        name=random_name,
        prompt="version 2",
        tags=["v2"],
    )

    latest_prompt = client.prompts.get(name=random_name)
    assert latest_prompt is not None
    assert latest_prompt.prompt == "version 2"


def test_tag_prompt_commit(client: Judgeval, random_name: str):
    prompt = client.prompts.create(
        name=random_name,
        prompt="test prompt content",
        tags=["original-tag"],
    )
    commit_id = prompt.commit_id

    new_tag = f"dynamic-tag-{uuid.uuid4().hex[:8]}"
    tagged_prompt = client.prompts.tag(
        name=random_name,
        commit_id=commit_id,
        tags=[new_tag],
    )
    assert tagged_prompt is not None

    retrieved_by_new_tag = client.prompts.get(
        name=random_name,
        tag=new_tag,
    )
    assert retrieved_by_new_tag is not None
    assert retrieved_by_new_tag.commit_id == commit_id

    retrieved_by_original_tag = client.prompts.get(
        name=random_name,
        tag="original-tag",
    )
    assert retrieved_by_original_tag is not None
    assert retrieved_by_original_tag.commit_id == commit_id


def test_untag_prompt(client: Judgeval, random_name: str):
    prompt = client.prompts.create(
        name=random_name,
        prompt="test prompt content",
        tags=["tag-1", "tag-2"],
    )
    prompt2 = client.prompts.create(
        name=random_name,
        prompt="test prompt content 2",
        tags=["tag-3", "tag-4"],
    )
    untagged_prompt = client.prompts.untag(
        name=random_name,
        tags=["tag-1"],
    )
    new_prompt_single = client.prompts.get(
        name=random_name,
        commit_id=prompt.commit_id,
    )
    assert untagged_prompt is not None
    assert len(untagged_prompt) == 1
    assert untagged_prompt[0] == prompt.commit_id
    assert "tag-1" not in new_prompt_single.tags

    untagged_prompt2 = client.prompts.untag(
        name=random_name,
        tags=["tag-2", "tag-3", "tag-4"],
    )

    new_prompt_double = client.prompts.get(
        name=random_name,
        commit_id=prompt.commit_id,
    )
    new_prompt_double_2 = client.prompts.get(
        name=random_name,
        commit_id=prompt2.commit_id,
    )
    assert untagged_prompt2 is not None
    assert len(untagged_prompt2) == 2
    assert set(untagged_prompt2) == {prompt.commit_id, prompt2.commit_id}
    assert "tag-2" not in new_prompt_double.tags
    assert "tag-3" not in new_prompt_double_2.tags
    assert "tag-4" not in new_prompt_double_2.tags


def test_overwrite_tag(client: Judgeval, random_name: str):
    prompt = client.prompts.create(
        name=random_name,
        prompt="test prompt content",
        tags=["tag-1"],
    )
    prompt2 = client.prompts.create(
        name=random_name,
        prompt="test prompt content 2",
    )
    tagged_prompt = client.prompts.tag(
        name=random_name,
        commit_id=prompt2.commit_id,
        tags=["tag-1"],
    )
    new_prompt1 = client.prompts.get(
        name=random_name,
        commit_id=prompt.commit_id,
    )
    new_prompt2 = client.prompts.get(
        name=random_name,
        commit_id=prompt2.commit_id,
    )
    assert tagged_prompt and tagged_prompt == prompt2.commit_id
    assert "tag-1" in new_prompt2.tags
    assert "tag-1" not in new_prompt1.tags


def test_list_prompts(client: Judgeval, random_name: str):
    prompt_name = random_name

    client.prompts.create(
        name=prompt_name,
        prompt="version 1",
        tags=["v1"],
    )

    client.prompts.create(
        name=prompt_name,
        prompt="version 2",
        tags=["v2"],
    )

    client.prompts.create(
        name=prompt_name,
        prompt="version 3",
        tags=["v3"],
    )

    prompt_list = client.prompts.list(name=prompt_name)
    assert prompt_list is not None
    assert len(prompt_list) == 3
    assert prompt_list[0].prompt == "version 3"
    assert prompt_list[1].prompt == "version 2"
    assert prompt_list[2].prompt == "version 1"


def test_multiple_tags_on_creation(client: Judgeval, random_name: str):
    tags = ["tag1", "tag2", "tag3"]
    prompt = client.prompts.create(
        name=random_name,
        prompt="test prompt with multiple tags",
        tags=tags,
    )
    commit_id = prompt.commit_id

    for tag in tags:
        retrieved = client.prompts.get(
            name=random_name,
            tag=tag,
        )
        assert retrieved is not None
        assert retrieved.commit_id == commit_id


def test_compile_prompt(client: Judgeval, random_name: str):
    prompt = client.prompts.create(
        name=random_name,
        prompt="test prompt {{name}}",
        tags=["tag-1"],
    )
    compiled_prompt = prompt.compile(name=random_name)
    assert compiled_prompt == f"test prompt {random_name}"


def test_get_nonexistent_prompt(client: Judgeval, random_name: str):
    result = client.prompts.get(
        name=random_name,
        tag="nonexistent_tag",
    )
    assert result is None

    result = client.prompts.get(
        name=random_name,
        commit_id="nonexistent_commit_id",
    )
    assert result is None

    result = client.prompts.get(name=random_name)
    assert result is None


def test_remove_nonexistent_tag(client: Judgeval, random_name: str):
    client.prompts.create(
        name=random_name,
        prompt="test prompt content",
        tags=["tag-1"],
    )
    with pytest.raises(JudgmentAPIError):
        client.prompts.untag(
            name=random_name,
            tags=["nonexistent_tag"],
        )


def test_tag_nonexistent_prompt(client: Judgeval, random_name: str):
    with pytest.raises(JudgmentAPIError):
        client.prompts.tag(
            name=random_name,
            commit_id="nonexistent_commit_id",
            tags=["tag-1"],
        )


def test_untag_nonexistent_prompt(client: Judgeval, random_name: str):
    with pytest.raises(JudgmentAPIError):
        client.prompts.untag(
            name=random_name,
            tags=["tag-1"],
        )


def test_tag_with_no_tags(client: Judgeval, random_name: str):
    with pytest.raises(JudgmentAPIError):
        client.prompts.tag(
            name=random_name,
            commit_id="nonexistent_commit_id",
            tags=[],
        )


def test_untag_with_no_tags(client: Judgeval, random_name: str):
    with pytest.raises(JudgmentAPIError):
        client.prompts.untag(
            name=random_name,
            tags=[],
        )
