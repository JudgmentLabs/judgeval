from __future__ import annotations

from typing import TypedDict, List


class UntagPromptResponse(TypedDict):
    commit_ids: List[str]
