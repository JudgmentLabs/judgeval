from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ScoreType = Literal["numeric", "binary", "categorical"]


@dataclass
class AgentJudge:
    """A prompt-based Agent Judge stored on the Judgment platform.

    Agent Judges are LLM-driven scorers. The `prompt` field is the **rubric
    prompt** used by the agent-judge harness when scoring an output. Versions
    are managed implicitly — calling `.update()` writes a new minor version of
    the underlying prompt scorer (matching the default "save" flow in the UI).

    Attributes:
        judge_id: Unique judge identifier on the Judgment platform.
        name: Human-readable name of the judge (unique per project).
        prompt: Rubric prompt template used by the agent judge.
        model: LiteLLM model id driving the agent judge (e.g. `"gpt-5.2"`).
        score_type: One of `"numeric"`, `"binary"`, or `"categorical"`.
        description: Optional description stored on the scorer version.
        judge_description: Optional human-readable description shown in the UI.
        categories: Choice list for `categorical` judges
            (e.g. `[{"name": "good", "description": "..."}, ...]`).
        min_score: Lower bound for `numeric` judges (defaults to `0`).
        max_score: Upper bound for `numeric` judges (defaults to `1`).
        major_version: Latest major version of the underlying prompt scorer.
        minor_version: Latest minor version of the underlying prompt scorer.

    Examples:
        ```python
        client = Judgeval(project_name="my-project")
        judge = client.agent_judges.create(
            name="helpfulness",
            prompt="Score the assistant's helpfulness from 0 to 1.",
            model="gpt-5.2",
            score_type="numeric",
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
