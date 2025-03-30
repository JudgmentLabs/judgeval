from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any, Union

from judgeval.data import ScorerData, ProcessExample

@dataclass
class ScoringResult:
    """
    A ScoringResult contains the output of one or more scorers applied to a single example.
    Ie: One input, one actual_output, one expected_output, etc..., and 1+ scorer (Faithfulness, Hallucination, Summarization, etc...)

    Args:
        success (bool): Whether the evaluation was successful. 
                        This means that all scorers applied to this example returned a success.
        scorer_data (List[ScorerData]): The scorers data for the evaluated example
        input (Optional[str]): The input to the example
        actual_output (Optional[str]): The actual output of the example
        expected_output (Optional[str]): The expected output of the example
        context (Optional[List[str]]): The context of the example
        retrieval_context (Optional[List[str]]): The retrieval context of the example
        additional_metadata (Optional[Dict[str, Any]]): The additional metadata of the example
        tools_called (Optional[List[str]]): The tools called by the example
        expected_tools (Optional[List[str]]): The expected tools of the example
        trace_id (Optional[str]): The trace id of the example
        
    """
    # Fields for scoring outputs 
    success: bool  # used for unit testing
    scorers_data: Union[List[ScorerData], None]

    # Inputs from the original example
    input: Optional[str] = None
    actual_output: Optional[Union[str, List[str]]] = None
    expected_output: Optional[Union[str, List[str]]] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    trace_id: Optional[str] = None
    
    example_id: Optional[str] = None
    eval_run_name: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert the ScoringResult instance to a dictionary, properly serializing scorer_data."""
        return {
            "success": self.success,
            "scorers_data": [scorer_data.to_dict() for scorer_data in self.scorers_data] if self.scorers_data else None,
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
            "additional_metadata": self.additional_metadata,
            "tools_called": self.tools_called,
            "expected_tools": self.expected_tools,
            "trace_id": self.trace_id,
            "example_id": self.example_id
        }
    
    def __str__(self) -> str:
        return f"ScoringResult(\
            success={self.success}, \
            scorer_data={self.scorers_data}, \
            input={self.input}, \
            actual_output={self.actual_output}, \
            expected_output={self.expected_output}, \
            context={self.context}, \
            retrieval_context={self.retrieval_context}, \
            additional_metadata={self.additional_metadata}, \
            tools_called={self.tools_called}, \
            expected_tools={self.expected_tools}, \
            trace_id={self.trace_id})"


def generate_scoring_result(
    process_example: ProcessExample,
) -> ScoringResult:
    """
    Creates a final ScoringResult object for an evaluation run based on the results from a completed LLMApiTestCase.

    When an LLMTestCase is executed, it turns into an LLMApiTestCase and the progress of the evaluation run is tracked.
    At the end of the evaluation run, we create a TestResult object out of the completed LLMApiTestCase.
    """
    return ScoringResult(
        success=process_example.success,
        scorers_data=process_example.scorers_data,
        input=process_example.input,
        actual_output=process_example.actual_output,
        expected_output=process_example.expected_output,
        context=process_example.context,
        retrieval_context=process_example.retrieval_context,
        additional_metadata=process_example.additional_metadata,
        tools_called=process_example.tools_called,
        expected_tools=process_example.expected_tools,
        trace_id=process_example.trace_id
    )
