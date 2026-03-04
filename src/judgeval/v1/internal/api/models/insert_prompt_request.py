from __future__ import annotations

from typing import TypedDict, Optional, List
from typing_extensions import NotRequired


class InsertPromptRequest(TypedDict):
    name: str
    prompt: str
    tags: NotRequired[Optional[List[str]]]
