from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from .judgment_types import (
    OtelSpanDetailScores,
    OtelSpanDetail,
    OtelTraceListItem
)


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

    # Override scores to use TraceScore (which has data field) instead of OtelSpanListItemScores
    scores: Optional[List["TraceScore"]] = []
    # All spans in the trace
    spans: List[TraceSpan] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert Trace to dictionary."""
        return self.model_dump(exclude_none=True)

    def get_root_span(self) -> Optional[TraceSpan]:
        """Get the root span (span with no parent)."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None

    def get_spans_by_parent(self, parent_span_id: Optional[str]) -> List[TraceSpan]:
        """Get all spans that have the specified parent span ID."""
        return [span for span in self.spans if span.parent_span_id == parent_span_id]

    def get_span_by_id(self, span_id: str) -> Optional[TraceSpan]:
        """Get a specific span by its ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def __len__(self) -> int:
        """Return the number of spans in the trace."""
        return len(self.spans)

    def __iter__(self):
        """Iterate over spans in the trace."""
        return iter(self.spans)
