"""
Classes for representing examples in a dataset.
"""

from enum import Enum
from datetime import datetime
from judgeval.data.judgment_types import ExampleJudgmentType


class ExampleParams(str, Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"
    REASONING = "reasoning"
    ADDITIONAL_METADATA = "additional_metadata"


class Example(ExampleJudgmentType):
    example_id: str = ""

    def __init__(self, **data):
        if "created_at" not in data:
            data["created_at"] = datetime.now().isoformat()
        super().__init__(**data)
        self.example_id = None

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
            "created_at": self.created_at,
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
            f"created_at={self.created_at}, "
        )
