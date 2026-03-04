from __future__ import annotations

from typing import TypedDict, Optional
from typing_extensions import NotRequired

from .prompt_commit_info import PromptCommitInfo


class FetchPromptResponse(TypedDict):
    commit: NotRequired[Optional[PromptCommitInfo]]
