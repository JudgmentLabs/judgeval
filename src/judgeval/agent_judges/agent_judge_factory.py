from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from judgeval.agent_judges.agent_judge import AgentJudge, ReasoningLevel, ScoreType
from judgeval.internal.api import JudgmentSyncClient
from judgeval.internal.api.models import (
    CreateJudgeRequest,
    CreateJudgeRequestJudge,
    CreateJudgeRequestJudgeConfig,
    CreateJudgeRequestJudgeInitialVersion,
    CreateJudgeRequestJudgeInitialVersionOutput,
    GetJudgeRequestVersion,
    GetJudgeResponseJudge,
    UpdateJudgeRequestUpdate,
    UpdateJudgeRequestUpdateNewVersion,
    UpdateJudgeRequestUpdateNewVersion1,
    UpdateJudgeRequestUpdateNewVersion2,
    UpdateJudgeRequestUpdateNewVersion2Outputs,
    UpdateJudgeRequestUpdateNewVersion2Outputs1,
    UpdateJudgeRequestUpdateNewVersionConfig,
    UpdateJudgeRequestUpdateNewVersionTargetVersion,
)
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id

# The platform creates only categorical judges (judgment-mono#5377), so a
# binary judge is created as two categorical outputs.
_BINARY_OUTPUTS: List[CreateJudgeRequestJudgeInitialVersionOutput] = [
    {"name": "Pass", "description": "The output meets the rubric."},
    {"name": "Fail", "description": "The output does not meet the rubric."},
]

JudgeVersion = Union[Literal["latest", "production"], GetJudgeRequestVersion]
VersionConfig = UpdateJudgeRequestUpdateNewVersionConfig
TargetVersion = UpdateJudgeRequestUpdateNewVersionTargetVersion

# The update route's `newVersion` is a union the spec leaves unnamed, so the
# generator numbers its members. These are the prompt-judge ones, by score type.
BinaryVersion = UpdateJudgeRequestUpdateNewVersion
NumericVersion = UpdateJudgeRequestUpdateNewVersion1
CategoricalVersion = UpdateJudgeRequestUpdateNewVersion2
PromptJudgeVersion = Union[BinaryVersion, NumericVersion, CategoricalVersion]
# A categorical output: an existing one by ID, or a new one by name/description.
CategoryOutput = Union[
    UpdateJudgeRequestUpdateNewVersion2Outputs,
    UpdateJudgeRequestUpdateNewVersion2Outputs1,
]


class AgentJudgeFactory:
    """Create and update prompt-based Agent judges on the Judgment platform.

    Access this via `client.agent_judges` — you don't instantiate it directly.

    Examples:
        ```python
        client = Judgeval(project_name="my-project")

        judge = client.agent_judges.create(
            name="helpfulness",
            prompt="Did the assistant answer the user's question?",
            score_type="binary",
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
            score_type: `"categorical"`, or `"binary"` for a categorical judge
                with `Pass` / `Fail` outputs. New judges cannot be `"numeric"`;
                existing numeric judges can still be updated.
            model: Deprecated and ignored; the platform picks the model
                from `reasoning_level`.
            reasoning_level: `"lite"` (default), `"normal"`, or `"high"`.
            description: Description stored on the initial judge version.
            judge_description: Description shown in the UI.
            categories: Choice list for `categorical` judges (at least two).
            min_score: Lower score bound stored on the version.
            max_score: Upper score bound stored on the version.

        Returns:
            The created `AgentJudge`, or `None` if the project is unresolved.

        Raises:
            ValueError: If `score_type` is `"numeric"`.
        """
        if score_type == "numeric":
            raise ValueError(
                "New Agent judges must be categorical. Use "
                "score_type='categorical' with categories, or 'binary' for "
                "Pass / Fail."
            )
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

        initial_version: CreateJudgeRequestJudgeInitialVersion = {
            "scoreType": "categorical",
            "outputs": _BINARY_OUTPUTS
            if score_type == "binary"
            else [
                {"name": c["name"], "description": c["description"]}
                for c in categories or []
            ],
        }

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

        return _to_agent_judge(self._fetch_judge(project_id, response["judge_id"]))

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
                an existing choice keep it, and its description must match.
            min_score: New lower bound for `numeric` judges.
            max_score: New upper bound for `numeric` judges.
            source_major_version: Major version to copy unspecified fields
                from. Defaults to the latest version.
            source_minor_version: Minor version to copy unspecified fields
                from. Defaults to the latest version; required when
                `source_major_version` is an older major.
            target_major_version: Major version to write to. Defaults to
                the current latest major.
            target_minor_version: Minor version to write to. Defaults to
                `latest minor + 1` in the latest major, else 0.

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
            latest = self._fetch_judge(project_id, judge_id)
            source = self._fetch_source_version(
                project_id, judge_id, latest, source_major_version, source_minor_version
            )
            target = _resolve_target_version(
                latest, target_major_version, target_minor_version
            )
            config = _build_version_config(
                source,
                prompt=prompt,
                description=description,
                reasoning_level=reasoning_level,
                min_score=min_score,
                max_score=max_score,
            )
            update["newVersion"] = _build_prompt_version(
                source,
                target,
                _resolve_score_type(score_type, source),
                categories,
                config,
            )
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

        return _to_agent_judge(self._fetch_judge(project_id, judge_id, written))

    def _fetch_judge(
        self, project_id: str, judge_id: str, version: JudgeVersion = "latest"
    ) -> GetJudgeResponseJudge:
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

    def _fetch_source_version(
        self,
        project_id: str,
        judge_id: str,
        latest: GetJudgeResponseJudge,
        major: Optional[int],
        minor: Optional[int],
    ) -> GetJudgeResponseJudge:
        """The version to copy unspecified fields from; `latest` by default."""
        latest_major = int(latest["judge_major_version"])
        major = latest_major if major is None else major
        if minor is None:
            # The lookup route takes an exact version, so an older major's
            # latest minor can't be resolved.
            if major != latest_major:
                raise ValueError(
                    "source_minor_version is required when source_major_version "
                    "is not the latest major"
                )
            return latest
        return self._fetch_judge(project_id, judge_id, {"major": major, "minor": minor})


def _resolve_target_version(
    latest: GetJudgeResponseJudge, major: Optional[int], minor: Optional[int]
) -> TargetVersion:
    """The version to write: `latest minor + 1` in the latest major by default."""
    latest_major = int(latest["judge_major_version"])
    major = latest_major if major is None else major
    if minor is None:
        minor = int(latest["judge_minor_version"]) + 1 if major == latest_major else 0
    return {"major": major, "minor": minor}


def _build_version_config(
    source: GetJudgeResponseJudge,
    *,
    prompt: Optional[str],
    description: Optional[str],
    reasoning_level: Optional[ReasoningLevel],
    min_score: Optional[float],
    max_score: Optional[float],
) -> VersionConfig:
    """The requested rubric fields, the rest copied from `source`."""
    definition = source["definition"]
    return {
        "prompt": prompt or definition["prompt"],
        "description": definition["description"]
        if description is None
        else description,
        "reasoningLevel": reasoning_level or definition["reasoning_level"],
        "minScore": definition["min_score"] if min_score is None else min_score,
        "maxScore": definition["max_score"] if max_score is None else max_score,
    }


def _resolve_score_type(
    requested: Optional[ScoreType], source: GetJudgeResponseJudge
) -> ScoreType:
    """The judge's own score type; `binary` also names create()'s Pass/Fail
    categorical judges."""
    if requested is None or (
        requested == "binary" and source["score_type"] == "categorical"
    ):
        return source["score_type"]
    return requested


def _build_prompt_version(
    source: GetJudgeResponseJudge,
    target: TargetVersion,
    score_type: ScoreType,
    categories: Optional[List[Dict[str, Any]]],
    config: VersionConfig,
) -> PromptJudgeVersion:
    """A complete new version; monitoring settings and action groups come from
    `source`."""
    version: BinaryVersion = {
        "targetVersion": target,
        "promoteToProduction": True,
        "judgeType": "prompt",
        "config": config,
        "scoreType": "binary",
        "advancedSettings": {
            "online_environment_names": source["online_environment_names"],
            "online_evaluation_mode": source["online_evaluation_mode"],
            "online_sampling_rate": source["online_sampling_rate"],
            "online_span_triggers": source["online_span_triggers"],
            "online_session_scoring": source["online_session_scoring"],
            "session_evaluation_count_enabled": source[
                "session_evaluation_count_enabled"
            ],
            "session_evaluation_idle_enabled": source[
                "session_evaluation_idle_enabled"
            ],
            "session_evaluation_min_candidate_traces": source[
                "session_evaluation_min_candidate_traces"
            ],
            "session_evaluation_max_wait_minutes": source[
                "session_evaluation_max_wait_minutes"
            ],
        },
    }
    # Omitted, the route keeps the latest version's action groups.
    if "action_config" in source:
        version["actionConfig"] = source["action_config"]
    if score_type == "numeric":
        numeric: NumericVersion = {**version, "scoreType": "numeric"}
        return numeric
    if score_type == "categorical":
        categorical: CategoricalVersion = {
            **version,
            "scoreType": "categorical",
            "outputs": _build_category_outputs(source, categories),
        }
        return categorical
    return version


def _build_category_outputs(
    source: GetJudgeResponseJudge, categories: Optional[List[Dict[str, Any]]]
) -> List[CategoryOutput]:
    """Existing outputs by ID (all of them when `categories` is None), new
    names by name and description."""
    if categories is None:
        return [{"behaviorId": o["behavior_id"]} for o in source["outputs"]]
    existing = {o["name"]: o for o in source["outputs"]}
    outputs: List[CategoryOutput] = []
    for c in categories:
        output = existing.get(c["name"])
        if output is None:
            outputs.append({"name": c["name"], "description": c["description"]})
        elif output["description"] != c["description"]:
            raise ValueError(
                f"Category {c['name']!r} already exists with a different "
                "description; a new version cannot change it."
            )
        else:
            outputs.append({"behaviorId": output["behavior_id"]})
    return outputs


def _warn_if_model(model: Optional[str]) -> None:
    if model is not None:
        judgeval_logger.warning(
            "Agent judge `model` is ignored; the platform picks the model "
            "from `reasoning_level`."
        )


def _to_agent_judge(judge: GetJudgeResponseJudge) -> AgentJudge:
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
