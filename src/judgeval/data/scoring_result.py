from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, cast

from judgeval.internal.api.models import (
    TraceSpan,
    ExampleScoringResult,
    TraceScoringResult,
    ScoringResult as APIScoringResult,
)
from judgeval.data.example import Example
from judgeval.data.scorer_data import ScorerData


@dataclass(slots=True)
class ScoringResult:
    """The combined result of running scorers against a single example.

    Returned by `Evaluation.run()`. Check `success` to see if **all**
    scorers passed, or inspect `scorers_data` for per-scorer details.

    Attributes:
        success: True only if every scorer met its threshold.
        scorers_data: Per-scorer results (see `ScorerData`).
        data_object: The `Example` or `TraceSpan` that was scored.
        name: The evaluation run name.
        trace_id: Associated trace ID, if applicable.
        run_duration: How long the evaluation took (seconds).
        evaluation_cost: Total cost in USD.

    Examples:
        ```python
        results = evaluation.run(
            examples=examples,
            scorers=["faithfulness", "answer_relevancy"],
            eval_run_name="nightly",
        )
        for result in results:
            if not result.success:
                for scorer in result.scorers_data:
                    print(f"{scorer.name}: {scorer.score} - {scorer.reason}")
        ```
    """

    success: bool
    scorers_data: List[ScorerData]
    data_object: Union[TraceSpan, Example]
    name: Optional[str] = None
    trace_id: Optional[str] = None
    run_duration: Optional[float] = None
    evaluation_cost: Optional[float] = None

    def to_dict(self) -> APIScoringResult:
        scorers_list: List[Dict[str, Any]] = [s.to_dict() for s in self.scorers_data]
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
