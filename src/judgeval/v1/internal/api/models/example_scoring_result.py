from __future__ import annotations

from typing import TypedDict, Optional, List, Any, Dict
from typing_extensions import NotRequired

from .example import Example


class ExampleScoringResult(TypedDict):
    scorers_data: List[Dict[str, Any]]
    name: NotRequired[Optional[str]]
    data_object: Example
    trace_id: NotRequired[Optional[str]]
    run_duration: NotRequired[Optional[float]]
    evaluation_cost: NotRequired[Optional[float]]
