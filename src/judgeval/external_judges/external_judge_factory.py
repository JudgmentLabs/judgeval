from __future__ import annotations

from typing import List, Optional, Union

from judgeval.external_judges.external_judge import (
    ExternalJudge,
    ExternalJudgeOutput,
    ExternalJudgeTarget,
    ScoreType,
)
from judgeval.exceptions import JudgmentAPIError, map_judgment_api_error
from judgeval.internal.api import JudgmentSyncClient
from judgeval.internal.api.models import (
    CreateJudgeRequest,
    CreateJudgeRequestJudge3InitialVersion,
    CreateJudgeRequestJudge3InitialVersion1,
    CreateJudgeRequestJudge3InitialVersion2,
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
            target={"type": "trace", "trace_id": "<trace_id>"},
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
        outputs: Optional[List[ExternalJudgeOutput]] = None,
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
        if score_type not in ("binary", "numeric", "categorical"):
            raise ValueError("score_type must be binary, numeric, or categorical")
        name = name.strip()
        if outputs is not None:
            outputs = [{**output, "name": output["name"].strip()} for output in outputs]

        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        initial_version: Union[
            CreateJudgeRequestJudge3InitialVersion,
            CreateJudgeRequestJudge3InitialVersion1,
            CreateJudgeRequestJudge3InitialVersion2,
        ]
        if score_type == "categorical":
            if not outputs or len(outputs) < 2:
                raise ValueError(
                    "outputs must have at least 2 entries for a 'categorical' judge"
                )
            initial_version = {
                "scoreType": "categorical",
                "outputs": outputs,
            }
        else:
            if outputs is not None:
                raise ValueError(
                    f"outputs is only valid when score_type='categorical' (got {score_type!r})"
                )
            initial_version = (
                {"scoreType": "binary"}
                if score_type == "binary"
                else {"scoreType": "numeric"}
            )

        payload: CreateJudgeRequest = {
            "judge": {
                "name": name,
                "executionMode": "external",
                "initialVersion": initial_version,
            }
        }
        if judge_description is not None:
            payload["judge"]["judgeDescription"] = judge_description

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
            judge_id=response["judge_id"],
            judge_version_id=response["judge_version_id"],
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
        target: ExternalJudgeTarget,
        value: Union[bool, float, str],
        judge_id: Optional[str] = None,
        judge_name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        """Submit a customer score under any judge for a trace or session.

        Resubmitting for the same judge and target replaces its current customer
        score. To correct a value, submit again for that judge and target.
        Each submission, including a retry, writes a new result row.

        Exactly one of `judge_id` or `judge_name` must be provided; the
        score is attached to that judge's production version. `value`
        must match the judge's score type (`bool` for `binary`, `int`/
        `float` for `numeric`, or one of the judge's configured outputs'
        names for `categorical`).

        Args:
            target: A trace or session target with its corresponding ID.
            value: The externally computed score.
            judge_id: ID of the judge to submit under.
            judge_name: Name of the judge to submit under.
            reason: Optional free-text explanation for the score.

        Returns:
            The id of the persisted score result, or `None` if the
            project is unresolved.

        Raises:
            ValueError: If neither or both of `judge_id`/`judge_name` are
                provided, or the target is invalid.
            JudgmentValidationError: If the server rejects the value or target.

        Examples:
            ```python
            client.external_judges.submit_result(
                judge_name="human-thumbs-up",
                target={"type": "trace", "trace_id": "<trace_id>"},
                value=True,
                reason="Reviewer approved the response.",
            )
            ```
        """
        if (judge_id is None) == (judge_name is None):
            raise ValueError("Exactly one of judge_id or judge_name must be provided")

        target_key = "trace_id" if target.get("type") == "trace" else "session_id"
        if (
            target.get("type") not in ("trace", "session")
            or set(target) != {"type", target_key}
            or not isinstance(target.get(target_key), str)
            or not target.get(target_key)
        ):
            raise ValueError("target must contain exactly one trace or session ID")

        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        payload: SubmitExternalJudgeResult
        if judge_id is not None:
            payload = {"target": target, "value": value, "judge_id": judge_id}
        else:
            assert judge_name is not None
            payload = {"target": target, "value": value, "judge_name": judge_name}
        if reason is not None:
            payload["reason"] = reason

        try:
            response = self._client.post_projects_judge_results(
                project_id=project_id,
                payload=payload,
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e, f"Failed to submit external judge result: {e.detail}"
            ) from e

        return response["id"]
