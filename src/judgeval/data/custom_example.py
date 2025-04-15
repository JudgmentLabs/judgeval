from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Any
from uuid import uuid4

class CustomExample(BaseModel):
    input: Optional[Dict[str, Any]] = None
    actual_output: Optional[Dict[str, Any]] = None
    expected_output: Optional[Dict[str, Any]] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    name: Optional[str] = None
    example_id: str = Field(default_factory=lambda: str(uuid4()))
    example_index: Optional[int] = None
    timestamp: Optional[str] = None
    trace_id: Optional[str] = None