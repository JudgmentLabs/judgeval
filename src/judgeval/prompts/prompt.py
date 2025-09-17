from typing import List, Optional
import os
from judgeval.api import JudgmentSyncClient
from judgeval.exceptions import JudgmentAPIError
from dataclasses import dataclass, field
import re
from string import Template


def push_prompt(
    name: str,
    prompt: str,
    tags: List[str],
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
) -> tuple[str, Optional[str]]:
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        r = client.prompts_insert(
            payload={"name": name, "prompt": prompt, "tags": tags}
        )
        return r["commit_id"], r["parent_commit_id"]
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to save prompt: {e.detail}",
            response=e.response,
        )


def fetch_prompt(
    name: str,
    commit_id: Optional[str] = None,
    tag: Optional[str] = None,
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
):
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        prompt_config = client.prompts_fetch(name, commit_id, tag)
        return prompt_config
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to fetch prompt '{name}': {e.detail}",
            response=e.response,
        )


@dataclass
class Prompt:
    name: str
    prompt: str
    tags: List[str]
    commit_id: str
    parent_commit_id: Optional[str] = None
    _template: Template = field(init=False, repr=False)

    def __post_init__(self):
        template_str = re.sub(r"\{\{(\w+)\}\}", r"$\1", self.prompt)
        self._template = Template(template_str)

    @classmethod
    def create(cls, name: str, prompt: str, tags: Optional[List[str]] = None):
        if not tags:
            tags = []
        commit_id, parent_commit_id = push_prompt(name, prompt, tags)
        return cls(
            name=name,
            prompt=prompt,
            tags=tags,
            commit_id=commit_id,
            parent_commit_id=parent_commit_id,
        )

    @classmethod
    def get(cls, name: str, commit_id: Optional[str] = None, tag: Optional[str] = None):
        if commit_id is not None and tag is not None:
            raise ValueError(
                "You cannot fetch a prompt by both commit_id and tag at the same time"
            )
        prompt_config = fetch_prompt(name, commit_id, tag)
        return cls(
            name=prompt_config["name"],
            prompt=prompt_config["prompt"],
            tags=prompt_config["tags"],
            commit_id=prompt_config["commit_id"],
            parent_commit_id=prompt_config["parent_commit_id"],
        )

    def compile(self, **kwargs) -> str:
        try:
            return self._template.substitute(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable: {missing_var}")
