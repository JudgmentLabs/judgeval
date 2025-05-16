from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from judgeval.evaluation_run import EvaluationRun
import json
from datetime import datetime, timezone

class TraceSpan(BaseModel):
    span_id: str
    trace_id: str
    function: Optional[str] = None
    depth: int
    created_at: Optional[float] = None
    parent_span_id: Optional[str] = None
    span_type: Optional[str] = "span"
    inputs: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    duration: Optional[float] = None
    annotation: Optional[List[Dict[str, Any]]] = None
    evaluation_runs: Optional[List[EvaluationRun]] = []

    def model_dump(self, **kwargs):
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "depth": self.depth,
            "created_at": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "inputs": self._serialize_inputs(),
            "output": self._serialize_output(),
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
    
    def _serialize_inputs(self) -> dict:
        """Helper method to serialize input data safely."""
        if self.inputs is None:
            return {}
            
        serialized_inputs = {}
        for key, value in self.inputs.items():
            if isinstance(value, BaseModel):
                serialized_inputs[key] = value.model_dump()
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples of arguments
                serialized_inputs[key] = [
                    item.model_dump() if isinstance(item, BaseModel)
                    else None if not self._is_json_serializable(item)
                    else item
                    for item in value
                ]
            else:
                if self._is_json_serializable(value):
                    serialized_inputs[key] = value
                else:
                    serialized_inputs[key] = self.safe_stringify(value, self.function)
        return serialized_inputs

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
        
    def _serialize_output(self) -> Any:
        """Helper method to serialize output data safely using an iterative approach."""
        if self.output is None:
            return None
        
        if isinstance(self.output, (str, int, float, bool, type(None))):
            return self.output
            
        stack = [(self.output, None, None)] 
        result = None
        
        while stack:
            value, container, key_or_index = stack.pop()
            processed_value = None
            
            if isinstance(value, BaseModel):
                processed_value = value.model_dump()
            elif isinstance(value, dict):
                processed_value = {}
                for k, v in value.items():
                    stack.append((v, processed_value, k))
            elif isinstance(value, (list, tuple)):
                processed_value = [None] * len(value)
                for i, v in enumerate(reversed(value)):
                    stack.append((v, processed_value, len(value) - 1 - i))
            else:
                try:
                    json.dumps(value)
                    processed_value = value
                except (TypeError, OverflowError, ValueError):
                    processed_value = self.safe_stringify(value, self.function)
            
            if container is not None:
                if isinstance(container, dict):
                    container[key_or_index] = processed_value
                elif isinstance(container, list):
                    container[key_or_index] = processed_value
            else:
                result = processed_value
                
        return result

class Trace(BaseModel):
    trace_id: str
    name: str
    created_at: str
    duration: float
    entries: List[TraceSpan]
    overwrite: bool = False
    rules: Optional[Dict[str, Any]] = None
    has_notification: Optional[bool] = False
    