from __future__ import annotations

from typing import List, Optional, overload

from judgeval.logger import judgeval_logger
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.internal.api import JudgmentSyncClient
from judgeval.prompts.prompt import Prompt
from judgeval.utils.guards import expect_project_id


class PromptFactory:
    """Create, retrieve, tag, and list versioned prompt templates.

    Access this via `client.prompts` -- you don't instantiate it directly.

    Examples:
        ```python
        # Create a new prompt
        prompt = client.prompts.create(
            name="system-prompt",
            prompt="You are a {{tone}} assistant for {{product}}.",
            tags=["v1"],
        )

        # Retrieve by tag
        prompt = client.prompts.get(name="system-prompt", tag="production")

        # Compile with variables
        text = prompt.compile(tone="helpful", product="Acme Search")
        ```
    """

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
        """Create a new prompt version.

        If a prompt with this name already exists, a new version is created
        automatically (linked via `parent_commit_id`).

        Args:
            name: Prompt name.
            prompt: The template string. Use `{{variable}}` for placeholders.
            tags: Tags to attach to this version (e.g. `["staging"]`).

        Returns:
            The created `Prompt`, or `None` if the project is not resolved.

        Examples:
            ```python
            prompt = client.prompts.create(
                name="system-prompt",
                prompt="You are a {{tone}} assistant for {{product}}.",
                tags=["v2"],
            )
            ```
        """
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
        """Retrieve a prompt by name.

        By default returns the latest version. Pass `tag` or `commit_id` to
        pin to a specific version.

        Args:
            name: Prompt name.
            commit_id: Fetch a specific version by its commit ID.
            tag: Fetch the version with this tag (e.g. `"production"`).

        Returns:
            The `Prompt`, or `None` if not found.

        Examples:
            ```python
            # Latest version
            prompt = client.prompts.get(name="system-prompt")

            # Pinned to production tag
            prompt = client.prompts.get(name="system-prompt", tag="production")
            ```
        """
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
        """Tag a specific prompt version.

        Use tags to mark versions for deployment (e.g. `"production"`,
        `"staging"`) so you can retrieve them by tag later.

        Args:
            name: Prompt name.
            commit_id: The version to tag.
            tags: Tags to attach (e.g. `["production"]`).

        Returns:
            The commit ID of the tagged version, or `None` on failure.

        Examples:
            ```python
            client.prompts.tag(
                name="system-prompt",
                commit_id=prompt.commit_id,
                tags=["production"],
            )
            ```
        """
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
        """Remove tags from a prompt.

        Args:
            name: Prompt name.
            tags: Tags to remove.

        Returns:
            Commit IDs of the versions that were untagged, or `None` on failure.
        """
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
        """List all versions of a prompt.

        Returns versions in reverse chronological order.

        Args:
            name: Prompt name.

        Returns:
            All `Prompt` versions, or `None` on failure.

        Examples:
            ```python
            versions = client.prompts.list(name="system-prompt")
            for v in versions:
                print(f"{v.commit_id} tags={v.tags}")
            ```
        """
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
