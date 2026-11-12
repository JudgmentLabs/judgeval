from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict, Union


ScoreType = Literal["binary", "numeric", "categorical"]


class ExternalJudgeOutput(TypedDict):
    """One named categorical result accepted by an external judge."""

    name: str
    description: str


class ExternalJudgeTraceTarget(TypedDict):
    """A trace whose current external judge score will be replaced."""

    type: Literal["trace"]
    trace_id: str


class ExternalJudgeSessionTarget(TypedDict):
    """A session whose current external judge score will be replaced."""

    type: Literal["session"]
    session_id: str


ExternalJudgeTarget = Union[ExternalJudgeTraceTarget, ExternalJudgeSessionTarget]


@dataclass
class ExternalJudge:
    """Metadata for a created external judge.

    Returned by `client.external_judges.create()`. For creation and result
    submission, see [ExternalJudgeFactory](/sdk-reference/python/external_judges/external_judge_factory).

    Attributes:
        judge_id: Unique judge identifier on the Judgment platform.
        judge_version_id: Identifier of this judge's initial version.
        name: Human-readable name of the judge (unique per project).
        score_type: One of `"binary"`, `"numeric"`, or `"categorical"`.
        judge_description: Optional human-readable description shown in the UI.
        outputs: Choice list for `categorical` judges
            (e.g. `[{"name": "good", "description": "..."}, ...]`).
            `None` for `binary`/`numeric` judges.
        major_version: Major version of the judge (`0` for a freshly
            created judge).
        minor_version: Minor version of the judge (`0` for a freshly
            created judge).

    """

    judge_id: str
    judge_version_id: str
    name: str
    score_type: ScoreType
    judge_description: Optional[str] = None
    outputs: Optional[List[ExternalJudgeOutput]] = None
    major_version: int = 0
    minor_version: int = 0
