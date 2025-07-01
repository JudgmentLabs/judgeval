from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from judgeval.evaluation_run import EvaluationRun
from judgeval.data.tool import Tool
import json
import sys
from datetime import datetime, timezone


class TraceUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_tokens_cost_usd: Optional[float] = None
    completion_tokens_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    model_name: Optional[str] = None


class TraceSpan(BaseModel):
    span_id: str
    trace_id: str
    function: str
    depth: int
    created_at: Optional[Any] = None
    parent_span_id: Optional[str] = None
    span_type: Optional[str] = "span"
    inputs: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    usage: Optional[TraceUsage] = None
    duration: Optional[float] = None
    annotation: Optional[List[Dict[str, Any]]] = None
    evaluation_runs: Optional[List[EvaluationRun]] = []
    expected_tools: Optional[List[Tool]] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    has_evaluation: Optional[bool] = False
    agent_name: Optional[str] = None
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None

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
            "evaluation_runs": [run.model_dump() for run in self.evaluation_runs]
            if self.evaluation_runs
            else [],
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
        }

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


class Trace(BaseModel):
    trace_id: str
    name: str
    created_at: str
    duration: float
    trace_spans: List[TraceSpan]
    overwrite: bool = False
    offline_mode: bool = False
    rules: Dict[str, Any] = Field(default_factory=dict)
    has_notification: Optional[bool] = False
    customer_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
