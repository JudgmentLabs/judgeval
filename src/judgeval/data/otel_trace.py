from typing import List
from pydantic import BaseModel
from judgeval.data.judgment_types import OtelTraceSpan


class TraceSpanData(OtelTraceSpan):
    pass


class TraceData(BaseModel):
    trace_spans: List[TraceSpanData]
