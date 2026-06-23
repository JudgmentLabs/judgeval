from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass(slots=True)
class ScorerData:
    """The result from a single scorer for one example.

    Each `ScoringResult` contains a list of `ScorerData` -- one per scorer
    that was run. Inspect `value` for the scorer's output.

    Attributes:
        name: Scorer name (e.g. `"faithfulness"`).
        value: The scorer result. Binary scorers use `"Yes"` or `"No"`;
            categorical scorers use their selected category.
        score_type: Result type returned by the scorer.
        minimum_score_range: Lower bound of the scoring scale.
        maximum_score_range: Upper bound of the scoring scale.
        evaluation_model: The LLM used to produce the score.
        error: Error message if scoring failed.
        additional_metadata: Extra metadata from the scorer.
        id: Unique identifier for this result.
        success: Outcome of a client-side pass condition for the row this
            scorer result belongs to. `None` when no pass condition was
            evaluated.
    """

    name: str
    value: Optional[Union[str, float]] = None
    score_type: Optional[str] = None
    minimum_score_range: float = 0
    maximum_score_range: float = 1
    evaluation_model: Optional[str] = None
    error: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    success: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary, omitting None fields."""
        result: Dict[str, Any] = {
            "name": self.name,
        }
        if self.value is not None:
            result["value"] = self.value
        if self.score_type is not None:
            result["score_type"] = self.score_type
        if self.minimum_score_range is not None:
            result["minimum_score_range"] = self.minimum_score_range
        if self.maximum_score_range is not None:
            result["maximum_score_range"] = self.maximum_score_range
        if self.evaluation_model is not None:
            result["evaluation_model"] = self.evaluation_model
        if self.error is not None:
            result["error"] = self.error
        if self.additional_metadata:
            result["additional_metadata"] = self.additional_metadata
        if self.id is not None:
            result["id"] = self.id
        if self.success is not None:
            result["success"] = self.success
        return result
