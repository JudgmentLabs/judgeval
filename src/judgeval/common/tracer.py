"""
Tracing system for judgeval that allows for function tracing using decorators.
"""

import time
import functools
import requests
import uuid
from contextlib import contextmanager
from typing import Optional, Any, List, Literal, Tuple, Generator, TypeAlias, Union
from dataclasses import dataclass, field
from datetime import datetime 
from openai import OpenAI
from together import Together
from anthropic import Anthropic
from typing import Dict
import inspect
import asyncio
import json
import warnings
from pydantic import BaseModel
from http import HTTPStatus

from judgeval.constants import JUDGMENT_TRACES_SAVE_API_URL
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import APIJudgmentScorer, JudgevalScorer
from judgeval.data.result import ScoringResult

# Define type aliases for better code readability and maintainability
ApiClient: TypeAlias = Union[OpenAI, Together, Anthropic]  # Supported API clients
TraceEntryType = Literal['enter', 'exit', 'output', 'input', 'evaluation']  # Valid trace entry types
SpanType = Literal['span', 'tool', 'llm', 'evaluation']
@dataclass
class TraceEntry:
    """Represents a single trace entry with its visual representation.
    
    Visual representations:
    - enter: → (function entry)
    - exit: ← (function exit)
    - output: Output: (function return value)
    - input: Input: (function parameters)
    - evaluation: Evaluation: (evaluation results)
    """
    type: TraceEntryType
    function: str  # Name of the function being traced
    depth: int    # Indentation level for nested calls
    message: str  # Human-readable description
    timestamp: float  # Unix timestamp when entry was created
    duration: Optional[float] = None  # Time taken (for exit/evaluation entries)
    output: Any = None  # Function output value
    # Use field() for mutable defaults to avoid shared state issues
    inputs: dict = field(default_factory=dict)
    span_type: SpanType = "span"
    evaluation_result: Optional[List[ScoringResult]] = field(default=None)
    
    def print_entry(self):
        indent = "  " * self.depth
        if self.type == "enter":
            print(f"{indent}→ {self.function} (trace: {self.message})")
        elif self.type == "exit":
            print(f"{indent}← {self.function} ({self.duration:.3f}s)")
        elif self.type == "output":
            print(f"{indent}Output: {self.output}")
        elif self.type == "input":
            print(f"{indent}Input: {self.inputs}")
        elif self.type == "evaluation":
            print(f"{indent}Evaluation: {self.evaluation_result} ({self.duration:.3f}s)")
    
    def _serialize_inputs(self) -> dict:
        """Helper method to serialize input data safely.
        
        Returns a dict with serializable versions of inputs, converting non-serializable
        objects to None with a warning.
        """
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
                    warnings.warn(f"Input '{key}' for function {self.function} is not JSON serializable. Setting to None.")
                    serialized_inputs[key] = None
        return serialized_inputs

    def _is_json_serializable(self, obj: Any) -> bool:
        """Helper method to check if an object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError, ValueError):
            return False

    def to_dict(self) -> dict:
        """Convert the trace entry to a dictionary format for storage/transmission."""
        return {
            "type": self.type,
            "function": self.function,
            "depth": self.depth,
            "message": self.message,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "output": self._serialize_output(),
            "inputs": self._serialize_inputs(),
            "evaluation_result": [result.to_dict() for result in self.evaluation_result] if self.evaluation_result else None,
            "span_type": self.span_type
        }

    def _serialize_output(self) -> Any:
        """Helper method to serialize output data safely.
        
        Handles special cases:
        - Pydantic models are converted using model_dump()
        - Non-serializable objects return None with a warning
        """
        if isinstance(self.output, BaseModel):
            return self.output.model_dump()
        
        try:
            # Try to serialize the output to verify it's JSON compatible
            json.dumps(self.output)
            return self.output
        except (TypeError, OverflowError, ValueError):
            warnings.warn(f"Output for function {self.function} is not JSON serializable. Setting to None.")
            return None

class TraceClient:
    """Client for managing a single trace context"""
    def __init__(self, tracer, trace_id: str, name: str, project_name: str = "default_project", overwrite: bool = False):
        self.tracer = tracer
        self.trace_id = trace_id
        self.name = name
        self.project_name = project_name
        self.client: JudgmentClient = tracer.client
        self.entries: List[TraceEntry] = []
        self.start_time = time.time()
        self.span_type = None
        self._current_span: Optional[TraceEntry] = None
        self.overwrite = overwrite
        
    @contextmanager
    def span(self, name: str, span_type: SpanType = "span"):
        """Context manager for creating a trace span"""
        start_time = time.time()
        
        # Record span entry
        self.add_entry(TraceEntry(
            type="enter",
            function=name,
            depth=self.tracer.depth,
            message=name,
            timestamp=start_time,
            span_type=span_type
        ))
        
        self.tracer.depth += 1
        prev_span = self._current_span
        self._current_span = name
        
        try:
            yield self
        finally:
            self.tracer.depth -= 1
            duration = time.time() - start_time
            
            # Record span exit
            self.add_entry(TraceEntry(
                type="exit",
                function=name,
                depth=self.tracer.depth,
                message=f"← {name}",
                timestamp=time.time(),
                duration=duration,
                span_type=span_type
            ))
            self._current_span = prev_span
            
    async def async_evaluate(
        self,
        scorers: List[Union[APIJudgmentScorer, JudgevalScorer]],
        input: Optional[str] = None,
        actual_output: Optional[str] = None,
        expected_output: Optional[str] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        tools_called: Optional[List[str]] = None,
        expected_tools: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        log_results: Optional[bool] = True,
    ):
        start_time = time.time()  # Record start time
        example = Example(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            tools_called=tools_called,
            expected_tools=expected_tools,
            additional_metadata=additional_metadata,
            trace_id=self.trace_id
        )
        scoring_results = self.client.run_evaluation(
            examples=[example],
            scorers=scorers,
            model=model,
            metadata={},
            log_results=log_results,
            project_name=self.project_name,
            eval_run_name=(
                f"{self.name.capitalize()}-"
                f"{self._current_span}-"
                f"[{','.join(scorer.load_implementation().score_type.capitalize() for scorer in scorers)}]"
            ),
            override=self.overwrite
        )
        
        self.record_evaluation(scoring_results, start_time)  # Pass start_time to record_evaluation
            
    def record_evaluation(self, results: List[ScoringResult], start_time: float):
        """Record evaluation results for the current span"""
        if self._current_span:
            duration = time.time() - start_time  # Calculate duration from start_time
            
            self.add_entry(TraceEntry(
                type="evaluation",
                function=self._current_span,
                depth=self.tracer.depth,
                message=f"Evaluation results for {self._current_span}",
                timestamp=time.time(),
                evaluation_result=results,
                duration=duration,
                span_type="evaluation"
            ))

    def record_input(self, inputs: dict):
        """Record input parameters for the current span"""
        if self._current_span:
            self.add_entry(TraceEntry(
                type="input",
                function=self._current_span,
                depth=self.tracer.depth,
                message=f"Inputs to {self._current_span}",
                timestamp=time.time(),
                inputs=inputs,
                span_type=self.span_type
            ))

    async def _update_coroutine_output(self, entry: TraceEntry, coroutine: Any):
        """Helper method to update the output of a trace entry once the coroutine completes"""
        try:
            result = await coroutine
            entry.output = result
            return result
        except Exception as e:
            entry.output = f"Error: {str(e)}"
            raise

    def record_output(self, output: Any):
        """Record output for the current span"""
        if self._current_span:
            entry = TraceEntry(
                type="output",
                function=self._current_span,
                depth=self.tracer.depth,
                message=f"Output from {self._current_span}",
                timestamp=time.time(),
                output="<pending>" if inspect.iscoroutine(output) else output,
                span_type=self.span_type
            )
            self.add_entry(entry)
            
            if inspect.iscoroutine(output):
                # Create a task to update the output once the coroutine completes
                asyncio.create_task(self._update_coroutine_output(entry, output))

    def add_entry(self, entry: TraceEntry):
        """Add a trace entry to this trace context"""
        self.entries.append(entry)
        return self
        
    def print(self):
        """Print the complete trace with proper visual structure"""
        for entry in self.entries:
            entry.print_entry()
            
    def get_duration(self) -> float:
        """
        Get the total duration of this trace
        """
        return time.time() - self.start_time
    
    def condense_trace(self, entries: List[dict]) -> List[dict]:
        """
        Condenses trace entries into a single entry for each function call.
        """
        condensed = []
        active_functions = []  # Stack to track nested function calls
        function_entries = {}  # Store entries for each function

        for entry in entries:
            function = entry["function"]
            
            if entry["type"] == "enter":
                # Initialize new function entry
                function_entries[function] = {
                    "depth": entry["depth"],
                    "function": function,
                    "timestamp": entry["timestamp"],
                    "inputs": None,
                    "output": None,
                    "evaluation_result": None,
                    "span_type": entry.get("span_type", "span")
                }
                active_functions.append(function)
                
            elif entry["type"] == "exit" and function in active_functions:
                # Complete function entry
                current_entry = function_entries[function]
                current_entry["duration"] = entry["timestamp"] - current_entry["timestamp"]
                condensed.append(current_entry)
                active_functions.remove(function)
                del function_entries[function]
                
            elif function in active_functions:
                # Update existing function entry with additional data
                current_entry = function_entries[function]
                
                if entry["type"] == "input" and entry["inputs"]:
                    current_entry["inputs"] = entry["inputs"]
                    
                if entry["type"] == "output" and entry["output"]:
                    current_entry["output"] = entry["output"]
                    
                if entry["type"] == "evaluation" and entry["evaluation_result"]:
                    current_entry["evaluation_result"] = entry["evaluation_result"]

        # Sort by timestamp
        condensed.sort(key=lambda x: x["timestamp"])
        return condensed

    def save(self, empty_save: bool = False, overwrite: bool = False) -> Tuple[str, dict]:
        """
        Save the current trace to the database.
        Returns a tuple of (trace_id, trace_data) where trace_data is the trace data that was saved.
        """
        # Calculate total elapsed time
        total_duration = self.get_duration()
        
        raw_entries = [entry.to_dict() for entry in self.entries]
        condensed_entries = self.condense_trace(raw_entries)

        # Create trace document
        trace_data = {
            "trace_id": self.trace_id,
            "api_key": self.tracer.api_key,
            "name": self.name,
            "project_name": self.project_name,
            "created_at": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration": total_duration,
            "token_counts": {
                "prompt_tokens": 0,  # Dummy value
                "completion_tokens": 0,  # Dummy value
                "total_tokens": 0,  # Dummy value
            },  # TODO: Add token counts
            "entries": condensed_entries,
            "empty_save": empty_save,
            "overwrite": overwrite
        }

        # Save trace data by making POST request to API
        response = requests.post(
            JUDGMENT_TRACES_SAVE_API_URL,
            json=trace_data,
            headers={
                "Content-Type": "application/json",
            }
        )
        
        if response.status_code == HTTPStatus.BAD_REQUEST:
            raise ValueError(f"Failed to save trace data: Check your Trace name for conflicts, set overwrite=True to overwrite existing traces: {response.text}")
        elif response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to save trace data: {response.text}")
        
        return self.trace_id, trace_data

class Tracer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key: str):
        if not hasattr(self, 'initialized'):

            if not api_key:
                raise ValueError("Tracer must be configured with a Judgment API key")
            
            self.api_key: str = api_key
            self.client: JudgmentClient = JudgmentClient(judgment_api_key=api_key)
            self.depth: int = 0
            self._current_trace: Optional[str] = None
            self.initialized: bool = True
        
    @contextmanager
    def trace(self, name: str, project_name: str = "default_project", overwrite: bool = False) -> Generator[TraceClient, None, None]:
        """Start a new trace context using a context manager"""
        trace_id = str(uuid.uuid4())
        trace = TraceClient(self, trace_id, name, project_name=project_name, overwrite=overwrite)
        prev_trace = self._current_trace
        self._current_trace = trace
        
        # Automatically create top-level span
        with trace.span(name or "unnamed_trace") as span:
            try:
                # Save the trace to the database to handle Evaluations' trace_id referential integrity
                trace.save(empty_save=True, overwrite=overwrite)
                yield trace
            finally:
                self._current_trace = prev_trace
                
    def get_current_trace(self) -> Optional[TraceClient]:
        """
        Get the current trace context
        """
        return self._current_trace    

    def observe(self, func=None, *, name=None, span_type: SpanType = "span"):
        """
        Decorator to trace function execution with detailed entry/exit information.
        
        Args:
            func: The function to trace
            name: Optional custom name for the function
            span_type: The type of span to use for this observation (default: "span")
        """
        if func is None:
            return lambda f: self.observe(f, name=name, span_type=span_type)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if self._current_trace:
                    span_name = name or func.__name__
                    
                    with self._current_trace.span(span_name, span_type=span_type) as span:
                        # Set the span type
                        span.span_type = span_type
                        
                        # Record inputs
                        span.record_input({
                            'args': list(args),
                            'kwargs': kwargs
                        })
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Record output
                        span.record_output(result)
                        
                        return result
                
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self._current_trace:
                    span_name = name or func.__name__
                    
                    with self._current_trace.span(span_name, span_type=span_type) as span:
                        # Set the span type
                        span.span_type = span_type
                        
                        # Record inputs
                        span.record_input({
                            'args': list(args),
                            'kwargs': kwargs
                        })
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Record output
                        span.record_output(result)
                        
                        return result
                
                return func(*args, **kwargs)
            return wrapper

def wrap(client: Any) -> Any:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, and Anthropic clients.
    """
    tracer = Tracer._instance  # Get the global tracer instance
    
    # Get the appropriate configuration for this client type
    span_name, original_create = _get_client_config(client)
    
    def traced_create(*args, **kwargs):
        # Skip tracing if no active trace
        if not (tracer and tracer._current_trace):
            return original_create(*args, **kwargs)

        with tracer._current_trace.span(span_name, span_type="llm") as span:
            # Format and record the input parameters
            input_data = _format_input_data(client, **kwargs)
            span.record_input(input_data)
            
            # Make the actual API call
            response = original_create(*args, **kwargs)
            
            # Format and record the output
            output_data = _format_output_data(client, response)
            span.record_output(output_data)
            
            return response
            
    # Replace the original method with our traced version
    if isinstance(client, (OpenAI, Together)):
        client.chat.completions.create = traced_create
    elif isinstance(client, Anthropic):
        client.messages.create = traced_create
        
    return client

# Helper functions for client-specific operations

def _get_client_config(client: ApiClient) -> tuple[str, callable]:
    """Returns configuration tuple for the given API client.
    
    Args:
        client: An instance of OpenAI, Together, or Anthropic client
        
    Returns:
        tuple: (span_name, create_method)
            - span_name: String identifier for tracing
            - create_method: Reference to the client's creation method
            
    Raises:
        ValueError: If client type is not supported
    """
    if isinstance(client, OpenAI):
        return "OPENAI_API_CALL", client.chat.completions.create
    elif isinstance(client, Together):
        return "TOGETHER_API_CALL", client.chat.completions.create
    elif isinstance(client, Anthropic):
        return "ANTHROPIC_API_CALL", client.messages.create
    raise ValueError(f"Unsupported client type: {type(client)}")

def _format_input_data(client: ApiClient, **kwargs) -> dict:
    """Format input parameters based on client type.
    
    Extracts relevant parameters from kwargs based on the client type
    to ensure consistent tracing across different APIs.
    """
    if isinstance(client, (OpenAI, Together)):
        return {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
        }
    # Anthropic requires additional max_tokens parameter
    return {
        "model": kwargs.get("model"),
        "messages": kwargs.get("messages"),
        "max_tokens": kwargs.get("max_tokens")
    }

def _format_output_data(client: ApiClient, response: Any) -> dict:
    """Format API response data based on client type.
    
    Normalizes different response formats into a consistent structure
    for tracing purposes.
    
    Returns:
        dict containing:
            - content: The generated text
            - usage: Token usage statistics
    """
    if isinstance(client, (OpenAI, Together)):
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    # Anthropic has a different response structure
    return {
        "content": response.content[0].text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    }
