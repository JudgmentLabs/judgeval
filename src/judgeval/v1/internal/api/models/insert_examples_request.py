from __future__ import annotations

from typing import TypedDict, List

from .example import Example


class InsertExamplesRequest(TypedDict):
    examples: List[Example]
