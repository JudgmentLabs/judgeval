from __future__ import annotations

from typing import TypedDict, Optional, Any, Dict
from typing_extensions import NotRequired


class PromptScorer(TypedDict):
    id: str
    user_id: str
    organization_id: str
    name: str
    prompt: str
    threshold: float
    model: str
    options: NotRequired[Optional[Dict[str, Any]]]
    description: NotRequired[Optional[str]]
    created_at: NotRequired[Optional[str]]
    updated_at: NotRequired[Optional[str]]
    is_trace: NotRequired[Optional[bool]]
