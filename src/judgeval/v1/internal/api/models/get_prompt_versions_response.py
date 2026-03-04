from __future__ import annotations

from typing import TypedDict, List

from .prompt_commit_info import PromptCommitInfo


class GetPromptVersionsResponse(TypedDict):
    versions: List[PromptCommitInfo]
