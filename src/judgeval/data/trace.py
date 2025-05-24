from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from judgeval.evaluation_run import EvaluationRun
from judgeval.data.tool import Tool
import json
from datetime import datetime, timezone

class TraceSpan(BaseModel):
    span_id: str
    trace_id: str
    function: Optional[str] = None
    depth: int
    created_at: Optional[Any] = None
    parent_span_id: Optional[str] = None
    span_type: Optional[str] = "span"
    inputs: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    duration: Optional[float] = None
    annotation: Optional[List[Dict[str, Any]]] = None
    evaluation_runs: Optional[List[EvaluationRun]] = []
    expected_tools: Optional[List[Tool]] = None
    additional_metadata: Optional[Dict[str, Any]] = None

    def model_dump(self, **kwargs):
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "depth": self.depth,
            "created_at": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "inputs": self._serialize_value(self.inputs),
            "output": self._serialize_value(self.output),
            "error": self._serialize_value(self.error),
            "evaluation_runs": [run.model_dump() for run in self.evaluation_runs] if self.evaluation_runs else [],
            "parent_span_id": self.parent_span_id,
            "function": self.function,
            "duration": self.duration,
            "span_type": self.span_type
        }
    
    def print_span(self):
        """Print the span with proper formatting and parent relationship information."""
        indent = "  " * self.depth
        parent_info = f" (parent_id: {self.parent_span_id})" if self.parent_span_id else ""
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

        warnings.warn(
            f"Output for function {function_name} is not JSON serializable and could not be converted to string. Setting to None."
        )
        return None
        
    def _serialize_value(self, value: Any) -> Any:
        """Helper method to deep serialize a value safely supporting Pydantic Models / regular PyObjects."""
        if value is None:
            return None
            
        def serialize_value(value):
            if isinstance(value, BaseModel):
                return value.model_dump()
            elif isinstance(value, dict):
                # Recursively serialize dictionary values
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                # Recursively serialize list/tuple items
                return [serialize_value(item) for item in value]
            else:
                # Try direct JSON serialization first
                try:
                    json.dumps(value)
                    return value
                except (TypeError, OverflowError, ValueError):
                    # Fallback to safe stringification
                    return self.safe_stringify(value, self.function)

        # Start serialization with the top-level value
        return serialize_value(value)

class Trace(BaseModel):
    trace_id: str
    name: str
    created_at: str
    duration: float
    entries: List[TraceSpan]
    overwrite: bool = False
    offline_mode: bool = False
    rules: Optional[Dict[str, Any]] = None
    has_notification: Optional[bool] = False
    