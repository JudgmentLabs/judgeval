from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, cast

from judgeval.v1.internal.api.api_types import (
    TraceSpan,
    ScorerData as APIScorerData,
    ExampleScoringResult,
    TraceScoringResult,
    ScoringResult as APIScoringResult,
)
from judgeval.v1.data.example import Example
from judgeval.v1.data.scorer_data import ScorerData


@dataclass(slots=True)
class ScoringResult:
    success: bool
    scorers_data: List[ScorerData]
    data_object: Union[TraceSpan, Example]
    name: Optional[str] = None
    trace_id: Optional[str] = None
    run_duration: Optional[float] = None
    evaluation_cost: Optional[float] = None

    def to_dict(self) -> APIScoringResult:
        scorers_list: List[APIScorerData] = [s.to_dict() for s in self.scorers_data]
        result: Dict[str, Any] = {
            "success": self.success,
            "scorers_data": scorers_list,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.trace_id is not None:
            result["trace_id"] = self.trace_id
        if self.run_duration is not None:
            result["run_duration"] = self.run_duration
        if self.evaluation_cost is not None:
            result["evaluation_cost"] = self.evaluation_cost

        if isinstance(self.data_object, Example):
            result["data_object"] = self.data_object.to_dict()
            return cast(ExampleScoringResult, result)
        else:
            result["data_object"] = self.data_object
            return cast(TraceScoringResult, result)
