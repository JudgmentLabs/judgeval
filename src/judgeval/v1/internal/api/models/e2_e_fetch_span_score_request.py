from __future__ import annotations

from typing import TypedDict


class E2EFetchSpanScoreRequest(TypedDict):
    project_name: str
    trace_id: str
    span_id: str
