from datetime import datetime, timezone
from judgeval.data.judgment_types import (
    TraceUsage as JudgmentTraceUsage,
    TraceSpan as JudgmentTraceSpan,
    Trace as JudgmentTrace,
)
from judgeval.utils.serialize import json_encoder


class TraceUsage(JudgmentTraceUsage):
    pass


class TraceSpan(JudgmentTraceSpan):
    def model_dump(self, **kwargs):
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "created_at": datetime.fromtimestamp(
                self.created_at, tz=timezone.utc
            ).isoformat(),
            "inputs": json_encoder(self.inputs),
            "output": json_encoder(self.output),
            "error": json_encoder(self.error),
            "parent_span_id": self.parent_span_id,
            "function": self.function,
            "duration": self.duration,
            "span_type": self.span_type,
            "usage": self.usage.model_dump() if self.usage else None,
            "has_evaluation": self.has_evaluation,
            "agent_name": self.agent_name,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "additional_metadata": json_encoder(self.additional_metadata),
            "update_id": self.update_id,
        }


class Trace(JudgmentTrace):
    pass
