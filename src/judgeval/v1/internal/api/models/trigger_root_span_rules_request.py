from __future__ import annotations

from typing import TypedDict, List

from .trace_info import TraceInfo


class TriggerRootSpanRulesRequest(TypedDict):
    traces: List[TraceInfo]
