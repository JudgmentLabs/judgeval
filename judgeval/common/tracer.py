"""
Tracing system for judgeval that allows for function tracing using decorators.
"""

import time
import functools
import requests
import uuid
from contextlib import contextmanager
from typing import Optional, Any, List, Literal, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime 
from openai import OpenAI
from together import Together
from anthropic import Anthropic
from typing import Dict
import inspect
import asyncio

from judgeval.constants import JUDGMENT_TRACES_SAVE_API_URL
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import JudgmentScorer
from judgeval.data.result import ScoringResult

@dataclass
class TraceEntry:
    """
    Represents a single trace entry with its visual representation
    
    Each TraceEntry is a single line in the trace.
    The `type` field determines the visual representation of the entry.
        - `enter` is for when a function is entered, represented by `→`
        - `exit` is for when a function is exited, represented by `←`
        - `output` is for when a function outputs a value, represented by `Output:`
        - `input` is for function input parameters, represented by `Input:`
        - `evaluation` is for when a function is evaluated, represented by `Evaluation:`

    function: Name of the function being traced
    depth: Indentation level of this trace entry
    message: Additional message to include in the trace
    timestamp: Time when this trace entry was created
    duration: For 'exit' entries, how long the function took to execute
    output: For 'output' entries, the value that was output
    inputs: For 'input' entries, the inputs to the function
    evaluation_result: For 'evaluation' entries, the result of the evaluation
    """
    type: Literal['enter', 'exit', 'output', 'input', 'evaluation']
    function: str
    depth: int
    message: str
    timestamp: float
    duration: Optional[float] = None
    output: Any = None
    inputs: dict = None
    evaluation_result: Optional[List[ScoringResult]] = None
    
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
            print(f"{indent}Evaluation: {self.evaluation_result}")
    
    def to_dict(self):
        """Convert the trace entry to a dictionary format"""
        # Try to serialize output to check if it's JSON serializable
        try:
            # If output is a Pydantic model, serialize it
            from pydantic import BaseModel
            if isinstance(self.output, BaseModel):
                output = self.output.model_dump()
            else:
                # Test regular JSON serialization
                import json
                json.dumps(self.output)
                output = self.output
        except (TypeError, OverflowError, ValueError):
            import warnings
            warnings.warn(f"Output for function {self.function} is not JSON serializable. Setting to None.")
            output = None

        return {
            "type": self.type,
            "function": self.function,
            "depth": self.depth,
            "message": self.message,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "output": output,
            "inputs": self.inputs if self.inputs else None,
            "evaluation_result": [result.to_dict() for result in self.evaluation_result] if self.evaluation_result else None
        }

class TraceClient:
    """Client for managing a single trace context"""
    def __init__(self, tracer, trace_id: str, name: str):
        self.tracer = tracer
        self.trace_id = trace_id
        self.name = name
        self.client: JudgmentClient = tracer.client
        self.entries: List[TraceEntry] = []
        self.start_time = time.time()
        self._current_span = None
        
    @contextmanager
    def span(self, name: str):
        """Context manager for creating a trace span"""
        start_time = time.time()
        
        # Record span entry
        self.add_entry(TraceEntry(
            type="enter",
            function=name,
            depth=self.tracer.depth,
            message=name,
            timestamp=start_time
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
                duration=duration
            ))
            self._current_span = prev_span
            
    async def async_evaluate(
        self,
        input: Optional[str] = None,
        actual_output: Optional[str] = None,
        expected_output: Optional[str] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        tools_called: Optional[List[str]] = None,
        expected_tools: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        score_type: Optional[str] = None,
        threshold: Optional[float] = None,
        model: Optional[str] = None,
        log_results: Optional[bool] = False,
    ):
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
        scorer = JudgmentScorer(
            score_type=score_type,
            threshold=threshold
        )
        _, scoring_results = self.client.run_evaluation(
            examples=[example],
            scorers=[scorer],
            model=model,
            metadata={},
            log_results=log_results,
            project_name="TestSpanLevel",
            eval_run_name="TestSpanLevel",
        )
        
        self.record_evaluation(scoring_results)
            
    def record_evaluation(self, results: List[ScoringResult]):
        """Record evaluation results for the current span"""
        if self._current_span:
            self.add_entry(TraceEntry(
                type="evaluation",
                function=self._current_span,
                depth=self.tracer.depth,
                message=f"Evaluation results for {self._current_span}",
                timestamp=time.time(),
                evaluation_result=results
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
                inputs=inputs
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
                output="<pending>" if inspect.iscoroutine(output) else output
            )
            self.add_entry(entry)
            
            if inspect.iscoroutine(output):
                # Create a task to update the output once the coroutine completes
                asyncio.create_task(self._update_coroutine_output(entry, output))

    def add_entry(self, entry: TraceEntry):
        """Add a trace entry to this trace context"""
        self.entries.append(entry)
        return self
    
    def _sort_entries(self):
        return
        """Sort entries to ensure evaluations appear before exits"""
        i = 0
        while i < len(self.entries) - 1:
            curr = self.entries[i]
            next_entry = self.entries[i + 1]
            
            if (curr.type == "exit" and 
                next_entry.type == "evaluation"):
                # Update evaluation entry before swap
                next_entry.function = curr.function
                next_entry.depth = curr.depth + 1
                # Swap the entries
                self.entries[i], self.entries[i + 1] = self.entries[i + 1], self.entries[i]
            i += 1
        
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
        Condenses trace entries into a single entry for each function.
        
        Groups entries by function call and combines them into a single entry with:
        - depth: deepest depth for this function call
        - duration: time from first to last timestamp 
        - function: function name
        - inputs: non-None inputs
        - output: non-None outputs
        - evaluation_result: evaluation results
        - timestamp: first timestamp of the function call
        """
        condensed = []
        current_func = None
        current_entry = None

        for entry in entries:
            if entry["type"] == "enter":
                # Start of new function call
                current_func = entry["function"]
                current_entry = {
                    "depth": entry["depth"],
                    "function": entry["function"],
                    "timestamp": entry["timestamp"],
                    "inputs": None,
                    "output": None,
                    "evaluation_result": None
                }
            
            elif entry["type"] == "exit" and entry["function"] == current_func:
                # End of current function
                current_entry["duration"] = entry["timestamp"] - current_entry["timestamp"]
                condensed.append(current_entry)
                current_func = None
                current_entry = None
            
            elif current_func and entry["function"] == current_func:
                # Additional entries for current function
                if entry["depth"] > current_entry["depth"]:
                    current_entry["depth"] = entry["depth"]
                
                if entry["type"] == "input" and entry["inputs"]:
                    current_entry["inputs"] = entry["inputs"]
                    
                if entry["type"] == "output" and entry["output"]:
                    current_entry["output"] = entry["output"]
                    
                if entry["type"] == "evaluation" and entry["evaluation_result"]:
                    current_entry["evaluation_result"] = entry["evaluation_result"]

        return condensed

    def save(self) -> Tuple[str, dict]:
        """
        Save the current trace to the database.
        Returns a tuple of (trace_id, trace_data) where trace_data is the trace data that was saved.
        """
        # Calculate total elapsed time
        total_duration = self.get_duration()
        
        # Sort entries to ensure evaluations appear before exits
        # self._sort_entries()
        
        raw_entries = [entry.to_dict() for entry in self.entries]
        condensed_entries = self.condense_trace(raw_entries)

        # Create trace document
        trace_data = {
            "trace_id": self.trace_id,
            "api_key": self.tracer.api_key,
            "name": self.name,
            "created_at": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration": total_duration,
            "token_counts": {
                "prompt_tokens": 0,  # Dummy value
                "completion_tokens": 0,  # Dummy value
                "total_tokens": 0,  # Dummy value
            },  # TODO: Add token counts
            "entries": condensed_entries
        }

        # Save trace data by making POST request to API
        response = requests.post(
            JUDGMENT_TRACES_SAVE_API_URL,
            json=trace_data,
            headers={
                "Content-Type": "application/json",
            }
        )
        response.raise_for_status()
        
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
            
            self.api_key = api_key
            self.client = JudgmentClient(judgment_api_key=api_key)
            self.depth = 0
            self._current_trace: Optional[TraceClient] = None
            self.initialized = True
        
    @contextmanager
    def trace(self, name: str = None) -> Generator[TraceClient, None, None]:
        """Start a new trace context using a context manager"""
        trace_id = str(uuid.uuid4())
        trace = TraceClient(self, trace_id, name or "unnamed_trace")
        prev_trace = self._current_trace
        self._current_trace = trace
        
        # Automatically create top-level span
        with trace.span(name or "unnamed_trace") as span:
            try:
                # Save the trace to the database to handle Evaluations' trace_id referential integrity
                trace.save()
                self._current_trace.tasks = []  # Initialize empty task list
                yield trace
            finally:
                self._current_trace = prev_trace
                
    def get_current_trace(self) -> Optional[TraceClient]:
        """
        Get the current trace context
        """
        return self._current_trace    

    def observe(self, func=None, *, name=None):
        """
        Decorator to trace function execution with detailed entry/exit information.
        
        Args:
            func: The function to trace
            name: Optional custom name for the function
        """
        if func is None:
            return lambda f: self.observe(f, name=name)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if self._current_trace:
                    span_name = name or func.__name__
                    
                    with self._current_trace.span(span_name) as span:
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
                    
                    with self._current_trace.span(span_name) as span:
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
    
    if isinstance(client, OpenAI) or isinstance(client, Together):
        original_create = client.chat.completions.create
    elif isinstance(client, Anthropic):
        original_create = client.messages.create
    else:
        raise ValueError(f"Unsupported client type: {type(client)}")
    
    def traced_create(*args, **kwargs):
        if not (tracer and tracer._current_trace):
            return original_create(*args, **kwargs)

        # TODO: this is dangerous and prone to errors in future updates to how the class works.
        # If we add more model providers here, we need to add support for it here in the span names
        span_name = "OPENAI_API_CALL" if isinstance(client, OpenAI) else "TOGETHER_API_CALL" if isinstance(client, Together) else "ANTHROPIC_API_CALL"
        with tracer._current_trace.span(span_name) as span:
            # Record the input based on client type
            if isinstance(client, (OpenAI, Together)):
                input_data = {
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages"),
                }
            elif isinstance(client, Anthropic):
                input_data = {
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages"),
                    "max_tokens": kwargs.get("max_tokens")
                }
            
            span.record_input(input_data)
            
            # Make the API call
            response = original_create(*args, **kwargs)
            
            # Record the output based on client type
            if isinstance(client, (OpenAI, Together)):
                output_data = {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            
            elif isinstance(client, Anthropic):
                output_data = {
                    "content": response.content[0].text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    }
                }
            
            span.record_output(output_data)
            return response
            
    # Replace the original method with our traced version
    if isinstance(client, (OpenAI, Together)):
        client.chat.completions.create = traced_create
    elif isinstance(client, Anthropic):
        client.messages.create = traced_create
        
    return client
