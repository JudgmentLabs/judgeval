from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from judgeval.agent_judges.agent_judge import AgentJudge, ScoreType
from judgeval.internal.api import JudgmentSyncClient
from judgeval.internal.api.models import (
    SDKCreateAgentJudgeRequest,
    SDKUpdateAgentJudgeRequest,
)
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id


class AgentJudgeFactory:
    """Create and update prompt-based Agent Judges on the Judgment platform.

    Access this via `client.agent_judges` — you don't instantiate it directly.

    Examples:
        ```python
        client = Judgeval(project_name="my-project")

        judge = client.agent_judges.create(
            name="helpfulness",
            prompt="Rate the assistant's helpfulness on a scale of 0 to 1.",
            model="gpt-5.2",
            score_type="numeric",
        )

        client.agent_judges.update(
            judge_id=judge.judge_id,
            prompt="Updated rubric prompt.",
        )
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
        *,
        name: str,
        prompt: str,
        model: str,
        score_type: ScoreType,
        description: Optional[str] = None,
        judge_description: Optional[str] = None,
        categories: Optional[List[Dict[str, Any]]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> Optional[AgentJudge]:
        """Create a new Agent Judge.

        Args:
            name: Unique judge name within the project.
            prompt: Rubric prompt template used by the agent judge.
            model: LiteLLM model id (e.g. `"gpt-5.2"`).
            score_type: One of `"numeric"`, `"binary"`, or `"categorical"`.
            description: Description stored on the underlying scorer version.
            judge_description: Description shown in the UI.
            categories: Choice list for `categorical` judges.
            min_score: Lower bound for `numeric` judges (defaults to `0`).
            max_score: Upper bound for `numeric` judges (defaults to `1`).

        Returns:
            The created `AgentJudge`, or `None` if the project is unresolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        payload: SDKCreateAgentJudgeRequest = {
            "name": name,
            "prompt": prompt,
            "model": model,
            "score_type": score_type,
        }
        if description is not None:
            payload["description"] = description
        if judge_description is not None:
            payload["judge_description"] = judge_description
        if categories is not None:
            payload["categories"] = categories
        if min_score is not None:
            payload["min_score"] = min_score
        if max_score is not None:
            payload["max_score"] = max_score

        try:
            response = self._client.post_projects_judges(
                project_id=project_id,
                payload=payload,
            )
        except Exception as e:
            judgeval_logger.error(f"Failed to create agent judge: {e}")
            raise

        return AgentJudge(
            judge_id=response["judge_id"],
            name=name,
            prompt=prompt,
            model=model,
            score_type=score_type,
            description=description,
            judge_description=judge_description,
            categories=categories,
            min_score=min_score,
            max_score=max_score,
            major_version=0,
            minor_version=0,
        )

    def update(
        self,
        *,
        judge_id: str,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        score_type: Optional[ScoreType] = None,
        description: Optional[str] = None,
        judge_description: Optional[str] = None,
        categories: Optional[List[Dict[str, Any]]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        source_major_version: Optional[int] = None,
        source_minor_version: Optional[int] = None,
        target_major_version: Optional[int] = None,
        target_minor_version: Optional[int] = None,
    ) -> Optional[AgentJudge]:
        """Update an existing Agent Judge.

        Passing any of `prompt`, `model`, `categories`, `min_score`, or
        `max_score` writes a new version of the underlying prompt scorer.
        When `target_major_version` / `target_minor_version` are omitted,
        the server auto-bumps the latest version's minor by 1 — matching
        the UI's default "save" behaviour.

        Args:
            judge_id: ID of the judge to update.
            prompt: New rubric prompt template.
            model: New LiteLLM model id.
            score_type: New score type (`numeric`, `binary`, `categorical`).
            description: New scorer-version description.
            judge_description: New UI-facing description.
            categories: New choices for `categorical` judges.
            min_score: New lower bound for `numeric` judges.
            max_score: New upper bound for `numeric` judges.
            source_major_version: Major version to copy unspecified fields
                from. Defaults to the latest version.
            source_minor_version: Minor version to copy unspecified fields
                from. Defaults to the latest version.
            target_major_version: Major version to write to. Defaults to
                the current latest major.
            target_minor_version: Minor version to write to. Defaults to
                `latest minor + 1`.

        Returns:
            The updated `AgentJudge`, or `None` if the project is unresolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        payload: SDKUpdateAgentJudgeRequest = {}
        if prompt is not None:
            payload["prompt"] = prompt
        if model is not None:
            payload["model"] = model
        if score_type is not None:
            payload["score_type"] = score_type
        if description is not None:
            payload["description"] = description
        if judge_description is not None:
            payload["judge_description"] = judge_description
        if categories is not None:
            payload["categories"] = categories
        if min_score is not None:
            payload["min_score"] = min_score
        if max_score is not None:
            payload["max_score"] = max_score
        if source_major_version is not None:
            payload["source_major_version"] = source_major_version
        if source_minor_version is not None:
            payload["source_minor_version"] = source_minor_version
        if target_major_version is not None:
            payload["target_major_version"] = target_major_version
        if target_minor_version is not None:
            payload["target_minor_version"] = target_minor_version

        try:
            response = self._client.patch_projects_judges_by_judge_id(
                project_id=project_id,
                judge_id=judge_id,
                payload=payload,
            )
        except Exception as e:
            judgeval_logger.error(f"Failed to update agent judge: {e}")
            raise

        judge = response["judge"]
        return _agent_judge_from_detail(judge)


def _agent_judge_from_detail(detail: Dict[str, Any]) -> AgentJudge:
    return AgentJudge(
        judge_id=detail["id"],
        name=detail["name"],
        prompt=detail.get("prompt") or "",
        model=detail.get("model") or "",
        score_type=cast(ScoreType, detail["score_type"]),
        description=detail.get("description"),
        judge_description=detail.get("judge_description"),
        categories=detail.get("categories"),
        min_score=detail.get("min_score"),
        max_score=detail.get("max_score"),
        major_version=detail.get("major_version"),
        minor_version=detail.get("minor_version"),
    )
