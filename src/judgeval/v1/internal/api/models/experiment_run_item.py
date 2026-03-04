from __future__ import annotations

from typing import TypedDict, Optional, List, Any, Dict
from typing_extensions import NotRequired

from .experiment_scorer import ExperimentScorer


class ExperimentRunItem(TypedDict):
    organization_id: str
    experiment_run_id: str
    example_id: str
    data: Dict[str, Any]
    name: NotRequired[Optional[str]]
    created_at: str
    scorers: List[ExperimentScorer]
