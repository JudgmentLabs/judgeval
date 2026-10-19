from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from judgeval.agent_judges.agent_judge import AgentJudge, ReasoningLevel, ScoreType
from judgeval.internal.api import JudgmentSyncClient
from judgeval.internal.api.models import (
    CreateJudgeRequest,
    CreateJudgeRequestJudge,
    CreateJudgeRequestJudgeConfig,
    CreateJudgeRequestJudgeInitialVersion,
    CreateJudgeRequestJudgeInitialVersion1,
    CreateJudgeRequestJudgeInitialVersion2,
    JudgeLookupRequestVersion,
    JudgeLookupResponseJudge,
    UpdateJudgeRequestUpdate,
    UpdateJudgeRequestUpdateNewVersion,
    UpdateJudgeRequestUpdateNewVersion1,
    UpdateJudgeRequestUpdateNewVersion2,
    UpdateJudgeRequestUpdateNewVersion2Outputs,
    UpdateJudgeRequestUpdateNewVersion2Outputs1,
    UpdateJudgeRequestUpdateNewVersionAdvancedSettings,
    UpdateJudgeRequestUpdateNewVersionConfig,
    UpdateJudgeRequestUpdateNewVersionTargetVersion,
)
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id

JudgeVersion = Union[Literal["latest", "production"], JudgeLookupRequestVersion]
PromptVersion = Union[
    UpdateJudgeRequestUpdateNewVersion,
    UpdateJudgeRequestUpdateNewVersion1,
    UpdateJudgeRequestUpdateNewVersion2,
]


class AgentJudgeFactory:
    """Create and update prompt-based Agent judges on the Judgment platform.

    Access this via `client.agent_judges` — you don't instantiate it directly.

    Examples:
        ```python
        client = Judgeval(project_name="my-project")

        judge = client.agent_judges.create(
            name="helpfulness",
            prompt="Rate the assistant's helpfulness on a scale of 0 to 1.",
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
        score_type: ScoreType,
        model: Optional[str] = None,
        reasoning_level: Optional[ReasoningLevel] = None,
        description: Optional[str] = None,
        judge_description: Optional[str] = None,
        categories: Optional[List[Dict[str, Any]]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> Optional[AgentJudge]:
        """Create a new Agent judge.

        Args:
            name: Unique judge name within the project.
            prompt: Rubric prompt template used by the Agent judge.
            score_type: One of `"numeric"`, `"binary"`, or `"categorical"`.
            model: Deprecated and ignored; the platform picks the model
                from `reasoning_level`.
            reasoning_level: `"lite"` (default), `"normal"`, or `"high"`.
            description: Description stored on the initial judge version.
            judge_description: Description shown in the UI.
            categories: Choice list for `categorical` judges (at least two).
            min_score: Lower bound for `numeric` judges (defaults to `0`).
            max_score: Upper bound for `numeric` judges (defaults to `1`).

        Returns:
            The created `AgentJudge`, or `None` if the project is unresolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None
        _warn_if_model(model)

        config: CreateJudgeRequestJudgeConfig = {"prompt": prompt}
        if description is not None:
            config["description"] = description
        if reasoning_level is not None:
            config["reasoningLevel"] = reasoning_level
        if min_score is not None:
            config["minScore"] = min_score
        if max_score is not None:
            config["maxScore"] = max_score

        initial_version: Union[
            CreateJudgeRequestJudgeInitialVersion,
            CreateJudgeRequestJudgeInitialVersion1,
            CreateJudgeRequestJudgeInitialVersion2,
        ]
        if score_type == "categorical":
            initial_version = {
                "scoreType": "categorical",
                "outputs": [
                    {"name": c["name"], "description": c["description"]}
                    for c in categories or []
                ],
            }
        elif score_type == "binary":
            initial_version = {"scoreType": "binary"}
        else:
            initial_version = {"scoreType": "numeric"}

        judge: CreateJudgeRequestJudge = {
            "name": name,
            "judgeType": "prompt",
            "initialVersion": initial_version,
            "config": config,
        }
        if judge_description is not None:
            judge["judgeDescription"] = judge_description
        payload: CreateJudgeRequest = {"judge": judge}

        try:
            response = self._client.post_projects_judges(
                project_id=project_id,
                payload=payload,
            )
        except Exception as e:
            judgeval_logger.error(f"Failed to create Agent judge: {e}")
            raise

        return _agent_judge(self._lookup(project_id, response["judge_id"]))

    def update(
        self,
        *,
        judge_id: str,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_level: Optional[ReasoningLevel] = None,
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
        """Update an existing Agent judge.

        Passing any version field (`prompt`, `reasoning_level`, `score_type`,
        `description`, `categories`, `min_score`, `max_score`, or a
        source/target version) writes a new production version, copying
        unspecified fields from the source version. When
        `target_major_version` / `target_minor_version` are omitted, the
        latest version's minor is bumped by 1 — matching the UI's default
        "save" behaviour. The score type of a judge cannot change.

        Args:
            judge_id: ID of the judge to update.
            prompt: New rubric prompt template.
            model: Deprecated and ignored; the platform picks the model
                from `reasoning_level`.
            reasoning_level: New reasoning level.
            score_type: Must match the judge's existing score type.
            description: New judge-version description.
            judge_description: New UI-facing description.
            categories: New choices for `categorical` judges; names matching
                an existing choice keep it.
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
        _warn_if_model(model)

        update: UpdateJudgeRequestUpdate = {}
        if judge_description is not None:
            update["judgeDescription"] = judge_description
        written: JudgeVersion = "latest"
        version_fields = (
            prompt,
            reasoning_level,
            score_type,
            description,
            categories,
            min_score,
            max_score,
            source_major_version,
            source_minor_version,
            target_major_version,
            target_minor_version,
        )
        if any(field is not None for field in version_fields):
            has_source = (
                source_major_version is not None or source_minor_version is not None
            )
            source = self._lookup(
                project_id,
                judge_id,
                {"major": source_major_version or 0, "minor": source_minor_version or 0}
                if has_source
                else "latest",
            )
            latest = self._lookup(project_id, judge_id) if has_source else source
            latest_major = int(latest["judge_major_version"])
            target: UpdateJudgeRequestUpdateNewVersionTargetVersion = (
                {
                    "major": latest_major,
                    "minor": int(latest["judge_minor_version"]) + 1,
                }
                if target_major_version is None and target_minor_version is None
                else {
                    "major": latest_major
                    if target_major_version is None
                    else target_major_version,
                    "minor": target_minor_version or 0,
                }
            )
            new_version = _prompt_version(
                source,
                target,
                score_type or source["score_type"],
                categories,
                {
                    "prompt": prompt or source["definition"]["prompt"],
                    "description": source["definition"]["description"]
                    if description is None
                    else description,
                    "reasoningLevel": reasoning_level
                    or source["definition"]["reasoning_level"],
                    "minScore": source["definition"]["min_score"]
                    if min_score is None
                    else min_score,
                    "maxScore": source["definition"]["max_score"]
                    if max_score is None
                    else max_score,
                },
            )
            update["newVersion"] = new_version
            written = target

        try:
            self._client.patch_projects_judges_by_judge_id(
                project_id=project_id,
                judge_id=judge_id,
                payload={"update": update},
            )
        except Exception as e:
            judgeval_logger.error(f"Failed to update Agent judge: {e}")
            raise

        return _agent_judge(self._lookup(project_id, judge_id, written))

    def _lookup(
        self, project_id: str, judge_id: str, version: JudgeVersion = "latest"
    ) -> JudgeLookupResponseJudge:
        judge = self._client.post_projects_judges_by_judge_id_lookup(
            project_id=project_id,
            judge_id=judge_id,
            payload={"version": version},
        )["judge"]
        if judge["judge_type"] != "prompt":
            raise ValueError(
                f"Judge {judge_id} is a {judge['judge_type']} judge; only "
                "prompt-based agent judges are supported."
            )
        return judge


def _prompt_version(
    source: JudgeLookupResponseJudge,
    target: UpdateJudgeRequestUpdateNewVersionTargetVersion,
    score_type: ScoreType,
    categories: Optional[List[Dict[str, Any]]],
    config: UpdateJudgeRequestUpdateNewVersionConfig,
) -> PromptVersion:
    """A complete prompt-judge version, monitoring settings copied from `source`."""
    settings: UpdateJudgeRequestUpdateNewVersionAdvancedSettings = {
        "online_environment_names": source["online_environment_names"],
        "online_evaluation_mode": source["online_evaluation_mode"],
        "online_sampling_rate": source["online_sampling_rate"],
        "online_span_triggers": source["online_span_triggers"],
        "online_session_scoring": source["online_session_scoring"],
        "session_evaluation_count_enabled": source["session_evaluation_count_enabled"],
        "session_evaluation_idle_enabled": source["session_evaluation_idle_enabled"],
        "session_evaluation_min_candidate_traces": source[
            "session_evaluation_min_candidate_traces"
        ],
        "session_evaluation_max_wait_minutes": source[
            "session_evaluation_max_wait_minutes"
        ],
    }
    if score_type == "binary":
        return {
            "targetVersion": target,
            "promoteToProduction": True,
            "advancedSettings": settings,
            "judgeType": "prompt",
            "config": config,
            "scoreType": "binary",
        }
    if score_type == "numeric":
        return {
            "targetVersion": target,
            "promoteToProduction": True,
            "advancedSettings": settings,
            "judgeType": "prompt",
            "config": config,
            "scoreType": "numeric",
        }

    existing = {o["name"]: o["behavior_id"] for o in source["outputs"]}
    outputs: List[
        Union[
            UpdateJudgeRequestUpdateNewVersion2Outputs,
            UpdateJudgeRequestUpdateNewVersion2Outputs1,
        ]
    ] = []
    if categories is None:
        outputs = [{"behaviorId": o["behavior_id"]} for o in source["outputs"]]
    for c in categories or []:
        if c["name"] in existing:
            outputs.append({"behaviorId": existing[c["name"]]})
        else:
            outputs.append({"name": c["name"], "description": c["description"]})
    return {
        "targetVersion": target,
        "promoteToProduction": True,
        "advancedSettings": settings,
        "judgeType": "prompt",
        "config": config,
        "scoreType": "categorical",
        "outputs": outputs,
    }


def _warn_if_model(model: Optional[str]) -> None:
    if model is not None:
        judgeval_logger.warning(
            "Agent judge `model` is ignored; the platform picks the model "
            "from `reasoning_level`."
        )


def _agent_judge(judge: JudgeLookupResponseJudge) -> AgentJudge:
    definition = judge["definition"]
    return AgentJudge(
        judge_id=judge["id"],
        name=judge["name"],
        prompt=definition["prompt"],
        model=definition["model"],
        reasoning_level=definition["reasoning_level"],
        score_type=judge["score_type"],
        description=definition["description"],
        judge_description=judge["judge_description"],
        categories=[
            {"name": o["name"], "description": o["description"]}
            for o in judge["outputs"]
        ]
        if judge["score_type"] == "categorical"
        else None,
        min_score=definition["min_score"],
        max_score=definition["max_score"],
        major_version=int(judge["judge_major_version"]),
        minor_version=int(judge["judge_minor_version"]),
    )
