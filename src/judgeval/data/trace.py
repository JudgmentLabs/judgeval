from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, ConfigDict


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


class TraceScore(BaseModel):
    """Score information for a trace or span."""
    success: bool
    score: float
    reason: Optional[str] = None
    name: str
    data: Optional[Dict[str, Any]] = None


class TraceRule(BaseModel):
    """Rule that was triggered for a trace."""
    rule_id: str
    rule_name: str


class TraceSpan(BaseModel):
    """Individual span within a trace with complete telemetry data."""
    model_config = ConfigDict(extra="allow")

    organization_id: str
    project_id: str
    timestamp: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_state: Optional[str] = None
    span_name: Optional[str] = None
    span_kind: Optional[str] = None
    service_name: Optional[str] = None
    resource_attributes: Optional[Dict[str, Any]] = None
    span_attributes: Optional[Dict[str, Any]] = None
    duration: Optional[int] = None
    status_code: Optional[str] = None
    status_message: Optional[str] = None
    events: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    links: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    llm_cost: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    scores: Optional[List[TraceScore]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert TraceSpan to dictionary."""
        return self.model_dump(exclude_none=True)


class Trace(BaseModel):
    """Complete trace with metadata and all associated spans."""
    model_config = ConfigDict(extra="allow")

    # Trace-level metadata
    organization_id: str
    project_id: str
    trace_id: str
    timestamp: str
    duration: Optional[int] = None
    has_notification: Optional[bool] = None
    tags: Optional[List[str]] = None
    experiment_run_id: Optional[str] = None
    span_name: Optional[str] = None  # Root span name
    cumulative_llm_cost: Optional[float] = None
    error: Optional[Dict[str, Any]] = None
    scores: Optional[List[TraceScore]] = None
    customer_id: Optional[str] = None
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None
    annotation_count: Optional[int] = 0
    span_id: str  # Root span ID
    rules: Optional[List[TraceRule]] = None

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
