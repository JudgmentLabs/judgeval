from __future__ import annotations

from typing import TypedDict, Optional
from typing_extensions import NotRequired


class UploadCustomScorerRequest(TypedDict):
    scorer_name: str
    scorer_code: str
    requirements_text: str
    class_name: str
    overwrite: bool
    scorer_type: NotRequired[Optional[str]]
    response_type: str
    version: NotRequired[Optional[float]]
