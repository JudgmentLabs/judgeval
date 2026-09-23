from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


ScoreType = Literal["binary", "numeric", "categorical"]


@dataclass
class ExternalJudge:
    """A judge whose scores are computed outside the Judgment platform.

    External judges don't have an executable implementation on the
    platform — you create one to register its name and score shape, then
    call `.submit_result()` (as many times as you like) to attach scores
    your own evaluation code produced to a trace or session.

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

    judge_id: str
    judge_version_id: str
    name: str
    score_type: ScoreType
    judge_description: Optional[str] = None
    outputs: Optional[List[Dict[str, Any]]] = None
    major_version: int = 0
    minor_version: int = 0
