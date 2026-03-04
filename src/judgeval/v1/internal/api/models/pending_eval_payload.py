from __future__ import annotations

from typing import TypedDict, List, Any, Dict


class PendingEvalPayload(TypedDict):
    project_id: str
    eval_name: str
    judges: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    is_offline: bool
    is_behavior: bool
