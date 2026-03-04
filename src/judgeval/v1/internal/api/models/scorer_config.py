from __future__ import annotations

from typing import TypedDict, Optional, List, Any, Dict
from typing_extensions import NotRequired


class ScorerConfig(TypedDict):
    score_type: str
    name: str
    threshold: float
    model: NotRequired[Optional[str]]
    required_params: NotRequired[Optional[List[str]]]
    kwargs: NotRequired[Optional[Dict[str, Any]]]
    result_type: NotRequired[Optional[str]]
