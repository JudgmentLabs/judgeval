from __future__ import annotations

from typing import TypedDict


class TraceInfo(TypedDict):
    trace_id: str
    span_id: str
