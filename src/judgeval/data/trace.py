from typing import Any
import json
import sys
import threading
from datetime import datetime, timezone
from judgeval.data.judgment_types import (
    TraceUsageJudgmentType,
    TraceSpanJudgmentType,
    TraceJudgmentType,
)
from pydantic import BaseModel


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
            "inputs": self._serialize_value(self.inputs),
            "output": self._serialize_value(self.output),
            "error": self._serialize_value(self.error),
            "parent_span_id": self.parent_span_id,
            "function": self.function,
            "duration": self.duration,
            "span_type": self.span_type,
            "usage": self.usage.model_dump() if self.usage else None,
            "has_evaluation": self.has_evaluation,
            "agent_name": self.agent_name,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "additional_metadata": self._serialize_value(self.additional_metadata),
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

    def print_span(self):
        """Print the span with proper formatting and parent relationship information."""
        indent = "  " * self.depth
        parent_info = (
            f" (parent_id: {self.parent_span_id})" if self.parent_span_id else ""
        )
        print(f"{indent}→ {self.function} (id: {self.span_id}){parent_info}")

    def _is_json_serializable(self, obj: Any) -> bool:
        """Helper method to check if an object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError, ValueError):
            return False

    def safe_stringify(self, output, function_name):
        """
        Safely converts an object to a string or repr, handling serialization issues gracefully.
        """
        try:
            return str(output)
        except (TypeError, OverflowError, ValueError):
            pass

        try:
            return repr(output)
        except (TypeError, OverflowError, ValueError):
            pass

        return None

    def _serialize_value(self, value: Any) -> Any:
        """Helper method to deep serialize a value safely supporting Pydantic Models / regular PyObjects."""
        if value is None:
            return None

        recursion_limit = sys.getrecursionlimit()
        recursion_limit = int(recursion_limit * 0.75)

        def serialize_value(value, current_depth=0):
            try:
                if current_depth > recursion_limit:
                    return {"error": "max_depth_reached: " + type(value).__name__}

                if isinstance(value, BaseModel):
                    return value.model_dump()
                elif isinstance(value, dict):
                    # Recursively serialize dictionary values
                    return {
                        k: serialize_value(v, current_depth + 1)
                        for k, v in value.items()
                    }
                elif isinstance(value, (list, tuple)):
                    # Recursively serialize list/tuple items
                    return [serialize_value(item, current_depth + 1) for item in value]
                else:
                    # Try direct JSON serialization first
                    try:
                        json.dumps(value)
                        return value
                    except (TypeError, OverflowError, ValueError):
                        # Fallback to safe stringification
                        return self.safe_stringify(value, self.function)
                    except Exception:
                        return {"error": "Unable to serialize"}
            except Exception:
                return {"error": "Unable to serialize"}

        # Start serialization with the top-level value
        try:
            return serialize_value(value, current_depth=0)
        except Exception:
            return {"error": "Unable to serialize"}


class Trace(TraceJudgmentType):
    pass
