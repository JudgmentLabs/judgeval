from __future__ import annotations

from typing import TypedDict, Optional
from typing_extensions import NotRequired


class InsertPromptResponse(TypedDict):
    commit_id: str
    parent_commit_id: NotRequired[Optional[str]]
    created_at: str
