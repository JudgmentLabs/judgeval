from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from .judgment_types import (
    OtelSpanDetailScores,
    OtelSpanDetail,
    OtelTraceListItem,
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

    @classmethod
    def from_otel_span_detail(cls, span_detail: OtelSpanDetail) -> "TraceSpan":
        """Create TraceSpan from OtelSpanDetail, converting scores to TraceScore."""
        data = span_detail.model_dump()

        if "scores" in data and data["scores"]:
            data["scores"] = [TraceScore(**score) for score in data["scores"]]

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert TraceSpan to dictionary."""
        return self.model_dump(exclude_none=True)


class Trace(OtelTraceListItem):
    """Complete trace with metadata and all associated spans."""

    spans: List[TraceSpan] = []
    rules: Optional[List[TraceRule]] = []

    @classmethod
    def from_dataset_trace_with_spans(cls, dataset_trace: Any) -> "Trace":
        """Create Trace from DatasetTraceWithSpans (handles both API and judgment types)."""

        if hasattr(dataset_trace, "trace_detail"):
            trace_detail = dataset_trace.trace_detail
            spans_data = dataset_trace.spans
        else:
            trace_detail = dataset_trace.get("trace_detail", {})
            spans_data = dataset_trace.get("spans", [])

        if hasattr(trace_detail, "model_dump"):
            trace_data = trace_detail.model_dump()
        elif isinstance(trace_detail, dict):
            trace_data = trace_detail.copy()
        else:
            trace_data = dict(trace_detail)

        spans = []
        for span in spans_data:
            if hasattr(span, "model_dump"):
                spans.append(TraceSpan.from_otel_span_detail(span))
            else:
                # Handle dict spans
                span_data = dict(span) if not isinstance(span, dict) else span.copy()
                if "scores" in span_data and span_data["scores"]:
                    span_data["scores"] = [
                        TraceScore(**score)
                        if isinstance(score, dict)
                        else TraceScore(**score.model_dump())
                        for score in span_data["scores"]
                    ]
                spans.append(TraceSpan(**span_data))

        rules = []
        if "rule_id" in trace_data and trace_data["rule_id"]:
            rules = [
                TraceRule(
                    rule_id=trace_data["rule_id"],
                    rule_name=f"Rule {trace_data['rule_id']}",
                )
            ]

        trace_data.pop("scores", [])
        trace_data.pop("rule_id", None)
        trace = cls(**trace_data)

        trace.spans = spans
        trace.rules = rules

        return trace

    def to_dict(self) -> Dict[str, Any]:
        """Convert Trace to dictionary."""
        return self.model_dump(exclude_none=True)

    def __len__(self) -> int:
        """Return the number of spans in the trace."""
        return len(self.spans)

    def __iter__(self):
        """Iterate over spans in the trace."""
        return iter(self.spans)
