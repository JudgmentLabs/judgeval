from __future__ import annotations

from typing import TypedDict


class TriggerRootSpanRulesResponse(TypedDict):
    success: bool
    queued_traces: float
