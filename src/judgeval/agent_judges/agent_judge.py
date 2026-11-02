from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ScoreType = Literal["numeric", "binary", "categorical"]
ReasoningLevel = Literal["lite", "normal", "high"]


@dataclass
class AgentJudge:
    """A prompt-based Agent judge stored on the Judgment platform.

    Agent judges are LLM-driven scorers. The `prompt` field is the **rubric
    prompt** used by the Agent judge harness when scoring an output. Versions
    are managed implicitly — calling `.update()` writes a new minor version of
    the judge (matching the default "save" flow in the UI).

    Attributes:
        judge_id: Unique judge identifier on the Judgment platform.
        name: Human-readable name of the judge (unique per project).
        prompt: Rubric prompt template used by the Agent judge.
        model: Model the platform resolved from `reasoning_level`.
        score_type: One of `"numeric"`, `"binary"`, or `"categorical"`.
        description: Optional description stored on the judge version.
        judge_description: Optional human-readable description shown in the UI.
        categories: Choice list for `categorical` judges
            (e.g. `[{"name": "good", "description": "..."}, ...]`).
        min_score: Lower bound for `numeric` judges (defaults to `0`).
        max_score: Upper bound for `numeric` judges (defaults to `1`).
        major_version: Major version this judge was read at.
        minor_version: Minor version this judge was read at.
        reasoning_level: One of `"lite"`, `"normal"`, or `"high"`.

    Examples:
        ```python
        client = Judgeval(project_name="my-project")
        judge = client.agent_judges.create(
            name="helpfulness",
            prompt="Did the assistant answer the user's question?",
            score_type="binary",
        )

        judge = client.agent_judges.update(
            judge_id=judge.judge_id,
            prompt="Updated rubric prompt.",
        )
        ```
    """

    judge_id: str
    name: str
    prompt: str
    model: str
    score_type: ScoreType
    description: Optional[str] = None
    judge_description: Optional[str] = None
    categories: Optional[List[Dict[str, Any]]] = field(default=None)
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    major_version: Optional[int] = None
    minor_version: Optional[int] = None
    reasoning_level: Optional[ReasoningLevel] = None
