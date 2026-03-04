from __future__ import annotations

from typing import TypedDict, List

from .example import Example


class CreateDatasetRequest(TypedDict):
    name: str
    dataset_kind: str
    examples: List[Example]
    overwrite: bool
