from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from .judgment_types import OtelSpanDetailScores, OtelSpanDetail, OtelTraceListItem


class TraceUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_tokens_cost_usd: Optional[float] = None
    completion_tokens_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    model_name: Optional[str] = None


class TraceScore(OtelSpanDetailScores):
    """Score information for a trace or span."""

    pass


class TraceRule(BaseModel):
    """Rule that was triggered for a trace."""

    rule_id: str
    rule_name: str


class TraceSpan(OtelSpanDetail):
    """Individual span within a trace with complete telemetry data."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert TraceSpan to dictionary."""
        return self.model_dump(exclude_none=True)


class Trace(OtelTraceListItem):
    """Complete trace with metadata and all associated spans."""

    scores: Optional[List["TraceScore"]] = []
    spans: List[TraceSpan] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert Trace to dictionary."""
        return self.model_dump(exclude_none=True)

    def __len__(self) -> int:
        """Return the number of spans in the trace."""
        return len(self.spans)

    def __iter__(self):
        """Iterate over spans in the trace."""
        return iter(self.spans)
