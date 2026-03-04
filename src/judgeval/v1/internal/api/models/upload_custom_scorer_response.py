from __future__ import annotations

from typing import TypedDict


class UploadCustomScorerResponse(TypedDict):
    scorer_name: str
    status: str
    message: str
