from pydantic import BaseModel
from typing import Optional, Dict, List


class GroundTruthExample(BaseModel):
    """
    GroundTruthExample is the atomic unit of a `Dataset`. It is essentially the same
    as an `Example`, but the `actual_output` field is optional to enable users to 
    run their workflow on the `input` field at test-time to evaluate their current 
    workflow's performance.
    """
    input: str
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    source_file: Optional[str] = None
    trace_id: Optional[str] = None

    def to_dict(self):
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
            "additional_metadata": self.additional_metadata,
            "comments": self.comments,
            "tools_called": self.tools_called,
            "expected_tools": self.expected_tools,
            "source_file": self.source_file,
            "trace_id": self.trace_id,
        }
    
    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"input={self.input}, "
            f"actual_output={self.actual_output}, "
            f"expected_output={self.expected_output}, "
            f"context={self.context}, "
            f"retrieval_context={self.retrieval_context}, "
            f"additional_metadata={self.additional_metadata}, "
            f"comments={self.comments}, "
            f"tools_called={self.tools_called}, "
            f"expected_tools={self.expected_tools}, "
            f"source_file={self.source_file}, "
            f"trace_id={self.trace_id}"
            f")"
        )