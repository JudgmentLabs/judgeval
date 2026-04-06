from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class ScorerData:
    """The result from a single scorer for one example.

    Each `ScoringResult` contains a list of `ScorerData` -- one per scorer
    that was run. Check `success` to see if the score met the threshold, and
    `reason` for the scorer's explanation.

    Attributes:
        name: Scorer name (e.g. `"faithfulness"`).
        threshold: Minimum score required for `success=True`.
        success: Whether the score met or exceeded the threshold.
        score: The numeric score (0.0 to 1.0 by default).
        minimum_score_range: Lower bound of the scoring scale.
        maximum_score_range: Upper bound of the scoring scale.
        reason: The scorer's explanation for the score.
        evaluation_model: The LLM used to produce the score.
        error: Error message if scoring failed.
        additional_metadata: Extra metadata from the scorer.
        id: Unique identifier for this result.
    """

    name: str
    threshold: float
    success: bool
    score: Optional[float] = None
    minimum_score_range: float = 0
    maximum_score_range: float = 1
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None
    error: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary, omitting None fields."""
        result: Dict[str, Any] = {
            "name": self.name,
            "threshold": self.threshold,
            "success": self.success,
        }
        if self.score is not None:
            result["score"] = self.score
        if self.minimum_score_range is not None:
            result["minimum_score_range"] = self.minimum_score_range
        if self.maximum_score_range is not None:
            result["maximum_score_range"] = self.maximum_score_range
        if self.reason is not None:
            result["reason"] = self.reason
        if self.evaluation_model is not None:
            result["evaluation_model"] = self.evaluation_model
        if self.error is not None:
            result["error"] = self.error
        if self.additional_metadata:
            result["additional_metadata"] = self.additional_metadata
        if self.id is not None:
            result["id"] = self.id
        return result
