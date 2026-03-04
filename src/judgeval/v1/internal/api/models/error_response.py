from __future__ import annotations

from typing import TypedDict, Optional
from typing_extensions import NotRequired


class ErrorResponse(TypedDict):
    error: str
    message: NotRequired[Optional[str]]
