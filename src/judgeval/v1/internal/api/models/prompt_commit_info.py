from __future__ import annotations

from typing import TypedDict, Optional, List
from typing_extensions import NotRequired


class PromptCommitInfo(TypedDict):
    name: str
    prompt: str
    tags: List[str]
    commit_id: str
    parent_commit_id: NotRequired[Optional[str]]
    created_at: str
    first_name: str
    last_name: str
    user_email: str
