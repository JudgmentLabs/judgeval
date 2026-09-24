from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast

from judgeval.external_judges.external_judge import ExternalJudge, ScoreType
from judgeval.exceptions import JudgmentAPIError, map_judgment_api_error
from judgeval.internal.api import JudgmentSyncClient
from judgeval.internal.api.models import (
    CreateExternalJudgeInput,
    SubmitExternalJudgeResult,
)
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id


class ExternalJudgeFactory:
    """Create external judges and submit their results on the Judgment platform.

    Access this via `client.external_judges` — you don't instantiate it
    directly.

    Examples:
        ```python
        client = Judgeval(project_name="my-project")

        judge = client.external_judges.create(
            name="human-thumbs-up",
            score_type="binary",
        )

        client.external_judges.submit_result(
            judge_id=judge.judge_id,
            trace_id="<trace_id>",
            value=True,
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
        score_type: ScoreType,
        judge_description: Optional[str] = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ExternalJudge]:
        """Create a new external judge and its first version.

        Args:
            name: Unique judge name within the project.
            score_type: One of `"binary"`, `"numeric"`, or `"categorical"`.
            judge_description: Description shown in the UI.
            outputs: Choice list for `categorical` judges (e.g.
                `[{"name": "good", "description": "..."}, {"name": "bad",
                "description": "..."}]`). Required (2 or more) when
                `score_type` is `"categorical"`; must be omitted otherwise.

        Returns:
            The created `ExternalJudge`, or `None` if the project is
            unresolved.

        Raises:
            ValueError: If `outputs` is missing/too short for a
                `categorical` judge, or provided for a non-categorical one.
            JudgmentConflictError: If a judge with this name already
                exists in the project.
            JudgmentValidationError: If the server rejects the judge
                configuration.

        Examples:
            ```python
            judge = client.external_judges.create(
                name="topic-classifier",
                score_type="categorical",
                outputs=[
                    {"name": "billing", "description": "Billing questions"},
                    {"name": "support", "description": "Support requests"},
                ],
            )
            ```
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        if score_type == "categorical":
            if not outputs or len(outputs) < 2:
                raise ValueError(
                    "outputs must have at least 2 entries for a 'categorical' judge"
                )
            initial_version: Dict[str, Any] = {
                "scoreType": "categorical",
                "outputs": outputs,
            }
        else:
            if outputs is not None:
                raise ValueError(
                    f"outputs is only valid when score_type='categorical' (got {score_type!r})"
                )
            initial_version = {"scoreType": score_type}

        payload: CreateExternalJudgeInput = {
            "name": name,
            "judgeType": "external",
            "initialVersion": initial_version,
        }
        if judge_description is not None:
            payload["judgeDescription"] = judge_description

        try:
            response = self._client.post_projects_judges(
                project_id=project_id,
                payload=payload,
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e, f"Failed to create external judge '{name}': {e.detail}"
            ) from e

        judgeval_logger.info(f"Created external judge {name}")
        return ExternalJudge(
            judge_id=response["judgeId"],
            judge_version_id=response["judgeVersionId"],
            name=name,
            score_type=score_type,
            judge_description=judge_description,
            outputs=outputs,
            major_version=0,
            minor_version=0,
        )

    def submit_result(
        self,
        *,
        trace_id: str,
        value: Union[bool, float, str],
        judge_id: Optional[str] = None,
        judge_name: Optional[str] = None,
        session_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        """Submit an externally computed score to a trace or session.

        Exactly one of `judge_id` or `judge_name` must be provided; the
        score is attached to that judge's production version. `value`
        must match the judge's score type (`bool` for `binary`, `int`/
        `float` for `numeric`, or one of the judge's configured outputs'
        names for `categorical`).

        Args:
            trace_id: ID of the trace to score.
            value: The externally computed score.
            judge_id: ID of the judge to submit under.
            judge_name: Name of the judge to submit under.
            session_id: If provided, scopes the result to this session
                (must match the trace's session) instead of the trace alone.
            reason: Optional free-text explanation for the score.

        Returns:
            The id of the persisted score result, or `None` if the
            project is unresolved.

        Raises:
            ValueError: If neither or both of `judge_id`/`judge_name` are
                provided.
            JudgmentValidationError: If `value`/`session_id` doesn't match
                the judge or trace.

        Examples:
            ```python
            client.external_judges.submit_result(
                judge_name="human-thumbs-up",
                trace_id="<trace_id>",
                value=True,
                reason="Reviewer approved the response.",
            )
            ```
        """
        if (judge_id is None) == (judge_name is None):
            raise ValueError("Exactly one of judge_id or judge_name must be provided")

        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        payload: Dict[str, Any] = {"trace_id": trace_id, "value": value}
        if judge_id is not None:
            payload["judge_id"] = judge_id
        else:
            payload["judge_name"] = judge_name
        if session_id is not None:
            payload["session_id"] = session_id
        if reason is not None:
            payload["reason"] = reason

        try:
            # SubmitExternalJudgeResult is a top-level discriminated union in
            # the OpenAPI spec (judge_id XOR judge_name); the client
            # generator can't expand top-level unions into a real TypedDict
            # (same limitation as the already-shipped ScoringResult model),
            # so it generates an empty stub. The payload above is correct
            # per the backend contract regardless.
            response = self._client.post_projects_judge_results(
                project_id=project_id,
                payload=cast(SubmitExternalJudgeResult, payload),
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e, f"Failed to submit external judge result: {e.detail}"
            ) from e

        return response["id"]
