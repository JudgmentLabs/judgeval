"""
Classes for representing examples in a dataset.
"""


from typing import TypeVar, Optional, Any, Dict, List
from uuid import uuid4
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import time


Input = TypeVar('Input')
Output = TypeVar('Output')

class ExampleParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"
    REASONING = "reasoning"


class Example(BaseModel):
    input: Input
    actual_output: Output
    expected_output: Optional[str] = None
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

    def __init__(self, **data):
        super().__init__(**data)
        if 'example_id' not in data:
            data['example_id'] = str(uuid4())
        # Set timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")


    def to_dict(self):
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
            "additional_metadata": self.additional_metadata,
            "tools_called": self.tools_called,
            "expected_tools": self.expected_tools,
            "name": self.name,
            "example_id": self.example_id,
            "example_index": self.example_index,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id
        }

    def __str__(self):
        return (
            f"Example(input={self.input}, "
            f"actual_output={self.actual_output}, "
            f"expected_output={self.expected_output}, "
            f"context={self.context}, "
            f"retrieval_context={self.retrieval_context}, "
            f"additional_metadata={self.additional_metadata}, "
            f"tools_called={self.tools_called}, "
            f"expected_tools={self.expected_tools}, "
            f"name={self.name}, "
            f"example_id={self.example_id}, "
            f"example_index={self.example_index}, "
            f"timestamp={self.timestamp}, "
            f"trace_id={self.trace_id})"
        )
