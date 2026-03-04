from __future__ import annotations

from typing import TypedDict


class DatasetInfo(TypedDict):
    dataset_id: str
    name: str
    created_at: str
    kind: str
    entries: float
    creator: str
