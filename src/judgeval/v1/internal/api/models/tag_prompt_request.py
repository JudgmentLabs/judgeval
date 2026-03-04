from __future__ import annotations

from typing import TypedDict, List


class TagPromptRequest(TypedDict):
    commit_id: str
    tags: List[str]
