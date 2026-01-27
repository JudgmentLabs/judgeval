from __future__ import annotations

from typing import List, Optional, overload

from judgeval.logger import judgeval_logger
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.prompts.prompt import Prompt
from judgeval.v1.utils import require_project_id


class PromptFactory:
    __slots__ = ("_client", "_default_project_id")

    def __init__(
        self, client: JudgmentSyncClient, default_project_id: Optional[str] = None
    ):
        self._client = client
        self._default_project_id = default_project_id

    def create(
        self,
        name: str,
        prompt: str,
        tags: Optional[List[str]] = None,
    ) -> Prompt:
        try:
            if tags is None:
                tags = []

            project_id = require_project_id(self._default_project_id)
            response = self._client.prompts_insert(
                {
                    "project_id": project_id,
                    "name": name,
                    "prompt": prompt,
                    "tags": tags,
                }
            )
            return Prompt(
                name=name,
                prompt=prompt,
                created_at=response["created_at"],
                tags=tags,
                commit_id=response["commit_id"],
                parent_commit_id=response.get("parent_commit_id"),
            )
        except Exception as e:
            judgeval_logger.error(f"Failed to create prompt: {str(e)}")
            raise

    @overload
    def get(
        self,
        /,
        *,
        name: str,
        commit_id: str,
    ) -> Optional[Prompt]: ...

    @overload
    def get(
        self,
        /,
        *,
        name: str,
        tag: str,
    ) -> Optional[Prompt]: ...

    @dont_throw
    def get(
        self,
        /,
        *,
        name: str,
        commit_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Optional[Prompt]:
        if commit_id is not None and tag is not None:
            judgeval_logger.error("Cannot fetch prompt by both commit_id and tag")
            return None

        project_id = require_project_id(self._default_project_id)

        response = self._client.prompts_fetch(
            project_id=project_id,
            name=name,
            commit_id=commit_id,
            tag=tag,
        )

        prompt_config = response.get("commit")
        if prompt_config is None:
            return None

        return Prompt(
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

    def tag(
        self,
        name: str,
        commit_id: str,
        tags: List[str],
    ) -> str:
        try:
            project_id = require_project_id(self._default_project_id)
            response = self._client.prompts_tag(
                {
                    "project_id": project_id,
                    "name": name,
                    "commit_id": commit_id,
                    "tags": tags,
                }
            )
            return response["commit_id"]
        except Exception as e:
            judgeval_logger.error(f"Failed to tag prompt: {str(e)}")
            raise

    def untag(
        self,
        name: str,
        tags: List[str],
    ) -> List[str]:
        try:
            project_id = require_project_id(self._default_project_id)
            response = self._client.prompts_untag(
                {
                    "project_id": project_id,
                    "name": name,
                    "tags": tags,
                }
            )
            return response["commit_ids"]
        except Exception as e:
            judgeval_logger.error(f"Failed to untag prompt: {str(e)}")
            raise

    def list(
        self,
        name: str,
    ) -> List[Prompt]:
        try:
            project_id = require_project_id(self._default_project_id)
            response = self._client.prompts_get_prompt_versions(
                project_id=project_id,
                name=name,
            )

            return [
                Prompt(
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
                for prompt_config in response["versions"]
            ]
        except Exception as e:
            judgeval_logger.error(f"Failed to list prompt versions: {str(e)}")
            raise
