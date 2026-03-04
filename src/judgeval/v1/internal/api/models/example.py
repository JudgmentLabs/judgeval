from __future__ import annotations

from typing import TypedDict, Optional
from typing_extensions import NotRequired


class Example(TypedDict):
    example_id: str
    created_at: str
    name: NotRequired[Optional[str]]
    trace_id: NotRequired[Optional[str]]
    span_id: NotRequired[Optional[str]]
