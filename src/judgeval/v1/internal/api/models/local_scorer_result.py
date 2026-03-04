from __future__ import annotations

from typing import TypedDict, List, Any, Dict

from .example import Example


class LocalScorerResult(TypedDict):
    scorers_data: List[Dict[str, Any]]
    data_object: Example
