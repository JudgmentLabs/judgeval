from typing import List, Optional, Dict
from judgeval.api import JudgmentSyncClient
from judgeval.exceptions import JudgmentAPIError
from judgeval.api.api_types import (
    PromptCommitInfo,
    PromptTagResponse,
    PromptUntagResponse,
    PromptVersionsResponse,
)
from dataclasses import dataclass, field
import re
from string import Template
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID
from judgeval.utils.project import _resolve_project_id


def push_prompt(
    project_name: str,
    name: str,
    prompt: str,
    tags: List[str],
    judgment_api_key: str = JUDGMENT_API_KEY,
    organization_id: str = JUDGMENT_ORG_ID,
) -> tuple[str, Optional[str], str]:
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        project_id = _resolve_project_id(
            project_name, judgment_api_key, organization_id
        )
        if not project_id:
            raise JudgmentAPIError(
                status_code=404,
                detail=f"Project '{project_name}' not found",
                response=None,  # type: ignore
            )
        r = client.prompts_insert(
            payload={
                "project_id": project_id,
                "name": name,
                "prompt": prompt,
                "tags": tags,
            }
        )
        return r["commit_id"], r.get("parent_commit_id"), r["created_at"]
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to save prompt: {e.detail}",
            response=e.response,
        )


def fetch_prompt(
    project_name: str,
    name: str,
    commit_id: Optional[str] = None,
    tag: Optional[str] = None,
    judgment_api_key: str = JUDGMENT_API_KEY,
    organization_id: str = JUDGMENT_ORG_ID,
) -> Optional[PromptCommitInfo]:
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        project_id = _resolve_project_id(
            project_name, judgment_api_key, organization_id
        )
        if not project_id:
            raise JudgmentAPIError(
                status_code=404,
                detail=f"Project '{project_name}' not found",
                response=None,  # type: ignore
            )
        prompt_config = client.prompts_fetch(
            name=name,
            project_id=project_id,
            commit_id=commit_id,
            tag=tag,
        )
        return prompt_config["commit"]
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to fetch prompt '{name}': {e.detail}",
            response=e.response,
        )


def tag_prompt(
    project_name: str,
    name: str,
    commit_id: str,
    tags: List[str],
    judgment_api_key: str = JUDGMENT_API_KEY,
    organization_id: str = JUDGMENT_ORG_ID,
) -> PromptTagResponse:
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        project_id = _resolve_project_id(
            project_name, judgment_api_key, organization_id
        )
        if not project_id:
            raise JudgmentAPIError(
                status_code=404,
                detail=f"Project '{project_name}' not found",
                response=None,  # type: ignore
            )
        prompt_config = client.prompts_tag(
            payload={
                "project_id": project_id,
                "name": name,
                "commit_id": commit_id,
                "tags": tags,
            }
        )
        return prompt_config
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to tag prompt '{name}': {e.detail}",
            response=e.response,
        )


def untag_prompt(
    project_name: str,
    name: str,
    tags: List[str],
    judgment_api_key: str = JUDGMENT_API_KEY,
    organization_id: str = JUDGMENT_ORG_ID,
) -> PromptUntagResponse:
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        project_id = _resolve_project_id(
            project_name, judgment_api_key, organization_id
        )
        if not project_id:
            raise JudgmentAPIError(
                status_code=404,
                detail=f"Project '{project_name}' not found",
                response=None,  # type: ignore
            )
        prompt_config = client.prompts_untag(
            payload={"project_id": project_id, "name": name, "tags": tags}
        )
        return prompt_config
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to untag prompt '{name}': {e.detail}",
            response=e.response,
        )


def list_prompt(
    project_name: str,
    name: str,
    judgment_api_key: str = JUDGMENT_API_KEY,
    organization_id: str = JUDGMENT_ORG_ID,
) -> PromptVersionsResponse:
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        project_id = _resolve_project_id(
            project_name, judgment_api_key, organization_id
        )
        if not project_id:
            raise JudgmentAPIError(
                status_code=404,
                detail=f"Project '{project_name}' not found",
                response=None,  # type: ignore
            )
        prompt_config = client.prompts_get_prompt_versions(
            project_id=project_id, name=name
        )
        return prompt_config
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to list prompt '{name}': {e.detail}",
            response=e.response,
        )


@dataclass
class Prompt:
    name: str
    prompt: str
    created_at: str
    tags: List[str]
    commit_id: str
    parent_commit_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    _template: Template = field(init=False, repr=False)

    def __post_init__(self):
        template_str = re.sub(r"\{\{([^}]+)\}\}", r"$\1", self.prompt)
        self._template = Template(template_str)

    @classmethod
    def create(
        cls,
        project_name: str,
        name: str,
        prompt: str,
        tags: Optional[List[str]] = None,
        judgment_api_key: str = JUDGMENT_API_KEY,
        organization_id: str = JUDGMENT_ORG_ID,
    ):
        if tags is None:
            tags = []
        commit_id, parent_commit_id, created_at = push_prompt(
            project_name, name, prompt, tags, judgment_api_key, organization_id
        )
        return cls(
            name=name,
            prompt=prompt,
            created_at=created_at,
            tags=tags,
            commit_id=commit_id,
            parent_commit_id=parent_commit_id,
        )

    @classmethod
    def get(
        cls,
        project_name: str,
        name: str,
        commit_id: Optional[str] = None,
        tag: Optional[str] = None,
        judgment_api_key: str = JUDGMENT_API_KEY,
        organization_id: str = JUDGMENT_ORG_ID,
    ):
        if commit_id is not None and tag is not None:
            raise ValueError(
                "You cannot fetch a prompt by both commit_id and tag at the same time"
            )
        prompt_config = fetch_prompt(
            project_name, name, commit_id, tag, judgment_api_key, organization_id
        )
        if prompt_config is None:
            raise JudgmentAPIError(
                status_code=404,
                detail=f"Prompt '{name}' not found in project '{project_name}'",
                response=None,  # type: ignore
            )
        return cls(
            name=prompt_config["name"],
            prompt=prompt_config["prompt"],
            created_at=prompt_config["created_at"],
            tags=prompt_config["tags"],
            commit_id=prompt_config["commit_id"],
            parent_commit_id=prompt_config.get("parent_commit_id"),
            metadata={
                "creator_first_name": prompt_config["first_name"],
                "creator_last_name": prompt_config["last_name"],
                "creator_email": prompt_config["user_email"],
            },
        )

    @classmethod
    def tag(
        cls,
        project_name: str,
        name: str,
        commit_id: str,
        tags: List[str],
        judgment_api_key: str = JUDGMENT_API_KEY,
        organization_id: str = JUDGMENT_ORG_ID,
    ):
        prompt_config = tag_prompt(
            project_name, name, commit_id, tags, judgment_api_key, organization_id
        )
        return prompt_config["commit_id"]

    @classmethod
    def untag(
        cls,
        project_name: str,
        name: str,
        tags: List[str],
        judgment_api_key: str = JUDGMENT_API_KEY,
        organization_id: str = JUDGMENT_ORG_ID,
    ):
        prompt_config = untag_prompt(
            project_name, name, tags, judgment_api_key, organization_id
        )
        return prompt_config["commit_ids"]

    @classmethod
    def list(
        cls,
        project_name: str,
        name: str,
        judgment_api_key: str = JUDGMENT_API_KEY,
        organization_id: str = JUDGMENT_ORG_ID,
    ):
        prompt_configs = list_prompt(
            project_name, name, judgment_api_key, organization_id
        )["versions"]
        return [
            cls(
                name=prompt_config["name"],
                prompt=prompt_config["prompt"],
                tags=prompt_config["tags"],
                created_at=prompt_config["created_at"],
                commit_id=prompt_config["commit_id"],
                parent_commit_id=prompt_config.get("parent_commit_id"),
                metadata={
                    "creator_first_name": prompt_config["first_name"],
                    "creator_last_name": prompt_config["last_name"],
                    "creator_email": prompt_config["user_email"],
                },
            )
            for prompt_config in prompt_configs
        ]

    def compile(self, **kwargs) -> str:
        try:
            return self._template.substitute(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable: {missing_var}")
