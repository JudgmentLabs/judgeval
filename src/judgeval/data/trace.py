from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class TraceSpan(BaseModel):
    span_id: str
    trace_id: str
    function: Optional[str] = None
    depth: int
    created_at: Optional[str] = None
    parent_span_id: Optional[str] = None
    span_type: Optional[str] = "span"
    inputs: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    duration: Optional[float] = None
    annotation: Optional[List[Dict[str, Any]]] = None

class Trace(BaseModel):
    trace_id: str
    name: str
    created_at: str
    duration: float
    entries: List[TraceSpan]
    overwrite: bool = False
    rules: Optional[Dict[str, Any]] = None
    has_notification: Optional[bool] = False
    