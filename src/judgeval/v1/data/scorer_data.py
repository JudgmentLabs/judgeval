from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from judgeval.v1.internal.api.api_types import ScorerData as APIScorerData


@dataclass(slots=True)
class ScorerData:
    name: str
    threshold: float
    success: bool
    score: Optional[float] = None
    minimum_score_range: float = 0
    maximum_score_range: float = 1
    reason: Optional[str] = None
    strict_mode: Optional[bool] = None
    evaluation_model: Optional[str] = None
    error: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> APIScorerData:
        result: APIScorerData = {
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
        if self.strict_mode is not None:
            result["strict_mode"] = self.strict_mode
        if self.evaluation_model is not None:
            result["evaluation_model"] = self.evaluation_model
        if self.error is not None:
            result["error"] = self.error
        if self.additional_metadata:
            result["additional_metadata"] = self.additional_metadata
        if self.id is not None:
            result["id"] = self.id
        return result
