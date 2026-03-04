from __future__ import annotations

from typing import TypedDict, List


class AddTraceTagsRequest(TypedDict):
    tags: List[str]
