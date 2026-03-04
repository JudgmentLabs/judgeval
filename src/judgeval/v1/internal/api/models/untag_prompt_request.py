from __future__ import annotations

from typing import TypedDict, List


class UntagPromptRequest(TypedDict):
    tags: List[str]
