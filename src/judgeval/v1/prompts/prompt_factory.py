from __future__ import annotations

from typing import List, Optional, overload

from judgeval.logger import judgeval_logger
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.prompts.prompt import Prompt
from judgeval.utils.guards import expect_project_id


class PromptFactory:
    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    def create(
        self,
        name: str,
        prompt: str,
        tags: Optional[List[str]] = None,
    ) -> Optional[Prompt]:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        try:
            if tags is None:
                tags = []

            response = self._client.post_projects_prompts(
                project_id=project_id,
                payload={
                    "name": name,
                    "prompt": prompt,
                    "tags": tags,
                },
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

        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        response = self._client.get_projects_prompts_by_name(
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
    ) -> Optional[str]:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        try:
            response = self._client.post_projects_prompts_by_name_tags(
                project_id=project_id,
                name=name,
                payload={
                    "commit_id": commit_id,
                    "tags": tags,
                },
            )
            return response["commit_id"]
        except Exception as e:
            judgeval_logger.error(f"Failed to tag prompt: {str(e)}")
            raise

    def untag(
        self,
        name: str,
        tags: List[str],
    ) -> Optional[List[str]]:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        try:
            response = self._client.delete_projects_prompts_by_name_tags(
                project_id=project_id,
                name=name,
                payload={
                    "tags": tags,
                },
            )
            return response["commit_ids"]
        except Exception as e:
            judgeval_logger.error(f"Failed to untag prompt: {str(e)}")
            raise

    def list(
        self,
        name: str,
    ) -> Optional[List[Prompt]]:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        try:
            response = self._client.get_projects_prompts_by_name_versions(
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
