import threading
from datetime import datetime, timezone
from judgeval.data.judgment_types import (
    TraceUsageJudgmentType,
    TraceSpanJudgmentType,
    TraceJudgmentType,
)
from judgeval.constants import SPAN_LIFECYCLE_END_UPDATE_ID
from judgeval.common.api.json_encoder import json_encoder


class TraceUsage(TraceUsageJudgmentType):
    pass


class TraceSpan(TraceSpanJudgmentType):
    def model_dump(self, **kwargs):
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "depth": self.depth,
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
            "class_name": self.class_name,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "additional_metadata": json_encoder(self.additional_metadata),
            "update_id": self.update_id,
        }

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize thread lock for thread-safe update_id increment
        self._update_id_lock = threading.Lock()

    def increment_update_id(self) -> int:
        """
        Thread-safe method to increment the update_id counter.
        Returns:
            int: The new update_id value after incrementing
        """
        with self._update_id_lock:
            self.update_id += 1
            return self.update_id

    def set_update_id_to_ending_number(
        self, ending_number: int = SPAN_LIFECYCLE_END_UPDATE_ID
    ) -> int:
        """
        Thread-safe method to set the update_id to a predetermined ending number.

        Args:
            ending_number (int): The number to set update_id to. Defaults to SPAN_LIFECYCLE_END_UPDATE_ID.

        Returns:
            int: The new update_id value after setting
        """
        with self._update_id_lock:
            self.update_id = ending_number
            return self.update_id

    def print_span(self):
        """Print the span with proper formatting and parent relationship information."""
        indent = "  " * self.depth
        parent_info = (
            f" (parent_id: {self.parent_span_id})" if self.parent_span_id else ""
        )
        print(f"{indent}→ {self.function} (id: {self.span_id}){parent_info}")


class Trace(TraceJudgmentType):
    pass
