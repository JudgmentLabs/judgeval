"""
Tracing system for judgeval that allows for function tracing using decorators.
"""
# Standard library imports
import asyncio
import functools
import inspect
import json
import os
import time
import uuid
import warnings
import contextvars
import sys
from contextlib import contextmanager, asynccontextmanager, AbstractAsyncContextManager, AbstractContextManager # Import context manager bases
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    AsyncGenerator,
    TypeAlias,
)
from rich import print as rprint
import types # <--- Add this import

# Third-party imports
import pika
import requests
from litellm import cost_per_token
from pydantic import BaseModel
from rich import print as rprint
from openai import OpenAI, AsyncOpenAI
from together import Together, AsyncTogether
from anthropic import Anthropic, AsyncAnthropic
from google import genai

# Local application/library-specific imports
from judgeval.constants import (
    JUDGMENT_TRACES_SAVE_API_URL,
    JUDGMENT_TRACES_FETCH_API_URL,
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_QUEUE,
    JUDGMENT_TRACES_DELETE_API_URL,
    JUDGMENT_PROJECT_DELETE_API_URL,
)
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import APIJudgmentScorer, JudgevalScorer
from judgeval.rules import Rule
from judgeval.evaluation_run import EvaluationRun
from judgeval.data.result import ScoringResult

# Standard library imports needed for the new class
import concurrent.futures
from collections.abc import Iterator, AsyncIterator # Add Iterator and AsyncIterator

# Define context variables for tracking the current trace and the current span within a trace
current_trace_var = contextvars.ContextVar('current_trace', default=None)
current_span_var = contextvars.ContextVar('current_span', default=None) # ContextVar for the active span name
in_traced_function_var = contextvars.ContextVar('in_traced_function', default=False) # Track if we're in a traced function

# Define type aliases for better code readability and maintainability
ApiClient: TypeAlias = Union[OpenAI, Together, Anthropic, AsyncOpenAI, AsyncAnthropic, AsyncTogether, genai.Client, genai.client.AsyncClient]  # Supported API clients
TraceEntryType = Literal['enter', 'exit', 'output', 'input', 'evaluation']  # Valid trace entry types
SpanType = Literal['span', 'tool', 'llm', 'evaluation', 'chain']
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
    span_id: str # Unique ID for this specific span instance
    depth: int    # Indentation level for nested calls
    created_at: float # Unix timestamp when entry was created, replacing the deprecated 'timestamp' field
    function: Optional[str] = None  # Name of the function being traced
    message: Optional[str] = None  # Human-readable description
    duration: Optional[float] = None  # Time taken (for exit/evaluation entries)
    trace_id: str = None # ID of the trace this entry belongs to
    output: Any = None  # Function output value
    # Use field() for mutable defaults to avoid shared state issues
    inputs: dict = field(default_factory=dict)
    span_type: SpanType = "span"
    evaluation_runs: List[Optional[EvaluationRun]] = field(default=None)
    parent_span_id: Optional[str] = None # ID of the parent span instance
    
    def print_entry(self):
        """Print a trace entry with proper formatting and parent relationship information."""
        indent = "  " * self.depth
        
        if self.type == "enter":
            # Format parent info if present
            parent_info = f" (parent_id: {self.parent_span_id})" if self.parent_span_id else ""
            print(f"{indent}→ {self.function} (id: {self.span_id}){parent_info} (trace: {self.message})")
        elif self.type == "exit":
            print(f"{indent}← {self.function} (id: {self.span_id}) ({self.duration:.3f}s)")
        elif self.type == "output":
            # Format output to align properly
            output_str = str(self.output)
            print(f"{indent}Output (for id: {self.span_id}): {output_str}")
        elif self.type == "input":
            # Format inputs to align properly
            print(f"{indent}Input (for id: {self.span_id}): {self.inputs}")
        elif self.type == "evaluation":
            for evaluation_run in self.evaluation_runs:
                print(f"{indent}Evaluation (for id: {self.span_id}): {evaluation_run.model_dump()}")
    
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
    
        warnings.warn(
            f"Output for function {function_name} is not JSON serializable and could not be converted to string. Setting to None."
        )
        return None

    def to_dict(self) -> dict:
        """Convert the trace entry to a dictionary format for storage/transmission."""
        return {
            "type": self.type,
            "function": self.function,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "depth": self.depth,
            "message": self.message,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "duration": self.duration,
            "output": self._serialize_output(),
            "inputs": self._serialize_inputs(),
            "evaluation_runs": [evaluation_run.model_dump() for evaluation_run in self.evaluation_runs] if self.evaluation_runs else [],
            "span_type": self.span_type,
            "parent_span_id": self.parent_span_id,
        }

    def _serialize_output(self) -> Any:
        """Helper method to serialize output data safely.
        
        Handles special cases:
        - Pydantic models are converted using model_dump()
        - We try to serialize into JSON, then string, then the base representation (__repr__)
        - Non-serializable objects return None with a warning
        """
        
        if isinstance(self.output, BaseModel):
            return self.output.model_dump()
        
        # NEW check: If output is the dict structure from our stream wrapper
        if isinstance(self.output, dict) and 'streamed' in self.output:
            # Assume it's already JSON-serializable (content is string, usage is dict or None)
            return self.output
        # NEW check: If output is the placeholder string before stream completes
        elif self.output == "<pending stream>":
             # Represent this state clearly in the serialized data
            return {"status": "pending stream"}

        try:
            # Try to serialize the output to verify it's JSON compatible
            json.dumps(self.output)
            return self.output
        except (TypeError, OverflowError, ValueError):
            return self.safe_stringify(self.output, self.function)
        

class TraceManagerClient:
    """
    Client for handling trace endpoints with the Judgment API
    

    Operations include:
    - Fetching a trace by id
    - Saving a trace
    - Deleting a trace
    """
    def __init__(self, judgment_api_key: str, organization_id: str, tracer: Optional["Tracer"] = None):
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id
        self.tracer = tracer

    def fetch_trace(self, trace_id: str):
        """
        Fetch a trace by its id
        """
        response = requests.post(
            JUDGMENT_TRACES_FETCH_API_URL,
            json={
                "trace_id": trace_id,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id
            },
            verify=True
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to fetch traces: {response.text}")
        
        return response.json()
    


    def save_trace(self, trace_data: dict):
        """
        Saves a trace to the Judgment Supabase and optionally to S3 if configured.

        Args:
            trace_data: The trace data to save
            NOTE we save empty traces in order to properly handle async operations; we need something in the DB to associate the async results with
        """
        # Save to Judgment API
        response = requests.post(
            JUDGMENT_TRACES_SAVE_API_URL,
            json=trace_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id
            },
            verify=True
        )
        
        if response.status_code == HTTPStatus.BAD_REQUEST:
            raise ValueError(f"Failed to save trace data: Check your Trace name for conflicts, set overwrite=True to overwrite existing traces: {response.text}")
        elif response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to save trace data: {response.text}")
        
        # If S3 storage is enabled, save to S3 as well
        if self.tracer and self.tracer.use_s3:
            try:
                s3_key = self.tracer.s3_storage.save_trace(
                    trace_data=trace_data,
                    trace_id=trace_data["trace_id"],
                    project_name=trace_data["project_name"]
                )
                print(f"Trace also saved to S3 at key: {s3_key}")
            except Exception as e:
                warnings.warn(f"Failed to save trace to S3: {str(e)}")
        
        if "ui_results_url" in response.json():
            pretty_str = f"\n🔍 You can view your trace data here: [rgb(106,0,255)][link={response.json()['ui_results_url']}]View Trace[/link]\n"
            rprint(pretty_str)

    def delete_trace(self, trace_id: str):
        """
        Delete a trace from the database.
        """
        response = requests.delete(
            JUDGMENT_TRACES_DELETE_API_URL,
            json={
                "trace_ids": [trace_id],
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id
            }
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to delete trace: {response.text}")
        
        return response.json()
    
    def delete_traces(self, trace_ids: List[str]):
        """
        Delete a batch of traces from the database.
        """
        response = requests.delete(
            JUDGMENT_TRACES_DELETE_API_URL,
            json={
                "trace_ids": trace_ids,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id
            }
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to delete trace: {response.text}")
        
        return response.json()
    
    def delete_project(self, project_name: str):
        """
        Deletes a project from the server. Which also deletes all evaluations and traces associated with the project.
        """
        response = requests.delete(
            JUDGMENT_PROJECT_DELETE_API_URL,
            json={
                "project_name": project_name,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id
            }
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to delete traces: {response.text}")
            
        return response.json()


class TraceClient:
    """Client for managing a single trace context"""
    
    def __init__(
        self,
        tracer: Optional["Tracer"],
        trace_id: Optional[str] = None,
        name: str = "default",
        project_name: str = "default_project",
        overwrite: bool = False,
        rules: Optional[List[Rule]] = None,
        enable_monitoring: bool = True,
        enable_evaluations: bool = True,
        parent_trace_id: Optional[str] = None,
        parent_name: Optional[str] = None
    ):
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.project_name = project_name
        self.overwrite = overwrite
        self.tracer = tracer
        self.rules = rules or []
        self.enable_monitoring = enable_monitoring
        self.enable_evaluations = enable_evaluations
        self.parent_trace_id = parent_trace_id
        self.parent_name = parent_name
        self.client: JudgmentClient = tracer.client
        self.entries: List[TraceEntry] = []
        self.start_time = time.time()
        self.trace_manager_client = TraceManagerClient(tracer.api_key, tracer.organization_id, tracer)
        self.visited_nodes = []
        self.executed_tools = []
        self.executed_node_tools = []
        self._span_depths: Dict[str, int] = {} # NEW: To track depth of active spans
    
    def get_current_span(self):
        """Get the current span from the context var"""
        return current_span_var.get()
    
    def set_current_span(self, span: Any):
        """Set the current span from the context var"""
        return current_span_var.set(span)
    
    def reset_current_span(self, token: Any):
        """Reset the current span from the context var"""
        return current_span_var.reset(token)
        
    @contextmanager
    def span(self, name: str, span_type: SpanType = "span"):
        """Context manager for creating a trace span, managing the current span via contextvars"""
        start_time = time.time()
        
        # Generate a unique ID for *this specific span invocation*
        span_id = str(uuid.uuid4())
        
        parent_span_id = current_span_var.get() # Get ID of the parent span from context var
        token = current_span_var.set(span_id) # Set *this* span's ID as the current one
        
        current_depth = 0
        if parent_span_id and parent_span_id in self._span_depths:
            current_depth = self._span_depths[parent_span_id] + 1
        
        self._span_depths[span_id] = current_depth # Store depth by span_id
            
        entry = TraceEntry(
            type="enter",
            function=name,
            span_id=span_id,
            trace_id=self.trace_id,
            depth=current_depth,
            message=name,
            created_at=start_time,
            span_type=span_type,
            parent_span_id=parent_span_id,
        )
        self.add_entry(entry)
        
        try:
            yield self
        finally:
            duration = time.time() - start_time
            exit_depth = self._span_depths.get(span_id, 0) # Get depth using this span's ID
            self.add_entry(TraceEntry(
                type="exit",
                function=name,
                span_id=span_id, # Use the same span_id for exit
                trace_id=self.trace_id, # Use the trace_id from the trace client
                depth=exit_depth, 
                message=f"← {name}",
                created_at=time.time(),
                duration=duration,
                span_type=span_type,
            ))
            # Clean up depth tracking for this span_id
            if span_id in self._span_depths:
                del self._span_depths[span_id]
            # Reset context var
            current_span_var.reset(token)

    def async_evaluate(
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
        log_results: Optional[bool] = True
    ):
        if not self.enable_evaluations:
            return
        
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
        try:
            # Load appropriate implementations for all scorers
            if not scorers:
                warnings.warn("No valid scorers available for evaluation")
                return
            
            # Prevent using JudgevalScorer with rules - only APIJudgmentScorer allowed with rules
            if self.rules and any(isinstance(scorer, JudgevalScorer) for scorer in scorers):
                raise ValueError("Cannot use Judgeval scorers, you can only use API scorers when using rules. Please either remove rules or use only APIJudgmentScorer types.")
            
        except Exception as e:
            warnings.warn(f"Failed to load scorers: {str(e)}")
            return
        
        # Combine the trace-level rules with any evaluation-specific rules)
        eval_run = EvaluationRun(
            organization_id=self.tracer.organization_id,
            log_results=log_results,
            project_name=self.project_name,
            eval_name=f"{self.name.capitalize()}-"
                f"{current_span_var.get()}-"
                f"[{','.join(scorer.score_type.capitalize() for scorer in scorers)}]",
            examples=[example],
            scorers=scorers,
            model=model,
            metadata={},
            judgment_api_key=self.tracer.api_key,
            override=self.overwrite,
            trace_span_id=current_span_var.get(),
            rules=self.rules # Use the combined rules
        )
        
        self.add_eval_run(eval_run, start_time)  # Pass start_time to record_evaluation
            
    def add_eval_run(self, eval_run: EvaluationRun, start_time: float):
        current_span_id = current_span_var.get()
        if current_span_id:
            duration = time.time() - start_time
            prev_entry = self.entries[-1] if self.entries else None
            # Determine function name based on previous entry or context var (less ideal)
            function_name = "unknown_function" # Default
            if prev_entry and prev_entry.span_type == "llm":
                 function_name = prev_entry.function
            else:
                 # Try to find the function name associated with the current span_id
                 for entry in reversed(self.entries):
                     if entry.span_id == current_span_id and entry.type == 'enter':
                         function_name = entry.function
                         break
            
            # Get depth for the current span
            current_depth = self._span_depths.get(current_span_id, 0)

            self.add_entry(TraceEntry(
                type="evaluation",
                function=function_name,
                span_id=current_span_id, # Associate with current span
                trace_id=self.trace_id, # Use the trace_id from the trace client
                depth=current_depth,
                message=f"Evaluation results for {function_name}",
                created_at=time.time(),
                evaluation_runs=[eval_run],
                duration=duration,
                span_type="evaluation"
            ))

    def record_input(self, inputs: dict):
        current_span_id = current_span_var.get()
        if current_span_id:
            entry_span_type = "span"
            current_depth = self._span_depths.get(current_span_id, 0)
            function_name = "unknown_function" # Default
            for entry in reversed(self.entries):
                 if entry.span_id == current_span_id and entry.type == 'enter':
                      entry_span_type = entry.span_type
                      function_name = entry.function
                      break

            self.add_entry(TraceEntry(
                type="input",
                function=function_name,
                span_id=current_span_id, # Use current span_id
                trace_id=self.trace_id, # Use the trace_id from the trace client
                depth=current_depth,
                message=f"Inputs to {function_name}",
                created_at=time.time(),
                inputs=inputs,
                span_type=entry_span_type,
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
        current_span_id = current_span_var.get()
        if current_span_id:
            entry_span_type = "span"
            current_depth = self._span_depths.get(current_span_id, 0)
            function_name = "unknown_function" # Default
            for entry in reversed(self.entries):
                 if entry.span_id == current_span_id and entry.type == 'enter':
                      entry_span_type = entry.span_type
                      function_name = entry.function
                      break

            entry = TraceEntry(
                type="output",
                function=function_name,
                span_id=current_span_id, # Use current span_id
                depth=current_depth,
                message=f"Output from {function_name}",
                created_at=time.time(),
                output="<pending>" if inspect.iscoroutine(output) else output,
                span_type=entry_span_type,
            )
            self.add_entry(entry)
            
            if inspect.iscoroutine(output):
                asyncio.create_task(self._update_coroutine_output(entry, output))
            
            # Return the created entry
            return entry

    def add_entry(self, entry: TraceEntry):
        """Add a trace entry to this trace context"""
        self.entries.append(entry)
        return self
        
    def print(self):
        """Print the complete trace with proper visual structure"""
        for entry in self.entries:
            entry.print_entry()
            
    def print_hierarchical(self):
        """Print the trace in a hierarchical structure based on parent-child relationships"""
        # First, build a map of spans
        spans = {}
        root_spans = []
        
        # Collect all enter events first
        for entry in self.entries:
            if entry.type == "enter":
                spans[entry.function] = {
                    "name": entry.function,
                    "depth": entry.depth,
                    "parent_id": entry.parent_span_id,
                    "children": []
                }
                
                # If no parent, it's a root span
                if not entry.parent_span_id:
                    root_spans.append(entry.function)
                elif entry.parent_span_id not in spans:
                    # If parent doesn't exist yet, temporarily treat as root
                    # (we'll fix this later)
                    root_spans.append(entry.function)
        
        # Build parent-child relationships
        for span_name, span in spans.items():
            parent = span["parent_id"]
            if parent and parent in spans:
                spans[parent]["children"].append(span_name)
                # Remove from root spans if it was temporarily there
                if span_name in root_spans:
                    root_spans.remove(span_name)
        
        # Now print the hierarchy
        def print_span(span_name, level=0):
            if span_name not in spans:
                return
                
            span = spans[span_name]
            indent = "  " * level
            parent_info = f" (parent_id: {span['parent_id']})" if span["parent_id"] else ""
            print(f"{indent}→ {span_name}{parent_info}")
            
            # Print children
            for child in span["children"]:
                print_span(child, level + 1)
        
        # Print starting with root spans
        print("\nHierarchical Trace Structure:")
        for root in root_spans:
            print_span(root)
            
    def get_duration(self) -> float:
        """
        Get the total duration of this trace
        """
        return time.time() - self.start_time
    
    def condense_trace(self, entries: List[dict]) -> List[dict]:
        """
        Condenses trace entries into a single entry for each span instance,
        preserving parent-child span relationships using span_id and parent_span_id.
        """
        spans_by_id: Dict[str, dict] = {}
        evaluation_runs: List[EvaluationRun] = []

        # First pass: Group entries by span_id and gather data
        for entry in entries:
            span_id = entry.get("span_id")
            if not span_id:
                continue # Skip entries without a span_id (should not happen)

            if entry["type"] == "enter":
                if span_id not in spans_by_id:
                    spans_by_id[span_id] = {
                        "span_id": span_id,
                        "function": entry["function"],
                        "depth": entry["depth"], # Use the depth recorded at entry time
                        "created_at": entry["created_at"],
                        "trace_id": entry["trace_id"],
                        "parent_span_id": entry.get("parent_span_id"),
                        "span_type": entry.get("span_type", "span"),
                        "inputs": None,
                        "output": None,
                        "evaluation_runs": [],
                        "duration": None
                    }
                # Handle potential duplicate enter events if necessary (e.g., log warning)

            elif span_id in spans_by_id:
                current_span_data = spans_by_id[span_id]
                
                if entry["type"] == "input" and entry["inputs"]:
                    # Merge inputs if multiple are recorded, or just assign
                    if current_span_data["inputs"] is None:
                        current_span_data["inputs"] = entry["inputs"]
                    elif isinstance(current_span_data["inputs"], dict) and isinstance(entry["inputs"], dict):
                        current_span_data["inputs"].update(entry["inputs"])
                    # Add more sophisticated merging if needed

                elif entry["type"] == "output" and "output" in entry:
                    current_span_data["output"] = entry["output"]

                elif entry["type"] == "evaluation" and entry.get("evaluation_runs"):
                    if current_span_data.get("evaluation_runs") is not None:
                        evaluation_runs.extend(entry["evaluation_runs"])

                elif entry["type"] == "exit":
                    if current_span_data["duration"] is None: # Calculate duration only once
                        start_time = datetime.fromisoformat(current_span_data.get("created_at", entry["created_at"]))
                        end_time = datetime.fromisoformat(entry["created_at"])
                        current_span_data["duration"] = (end_time - start_time).total_seconds()
                    # Update depth if exit depth is different (though current span() implementation keeps it same)
                    # current_span_data["depth"] = entry["depth"] 

        # Convert dictionary to a list initially for easier access
        spans_list = list(spans_by_id.values())

        # Build tree structure (adjacency list) and find roots
        children_map: Dict[Optional[str], List[dict]] = {}
        roots = []
        span_map = {span['span_id']: span for span in spans_list} # Map for quick lookup

        for span in spans_list:
            parent_id = span.get("parent_span_id")
            if parent_id is None:
                roots.append(span)
            else:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(span)

        # Sort roots by timestamp
        roots.sort(key=lambda x: datetime.fromisoformat(x.get("created_at", "1970-01-01T00:00:00")))

        # Perform depth-first traversal to get the final sorted list
        sorted_condensed_list = []
        visited = set() # To handle potential cycles, though unlikely with UUIDs

        def dfs(span_data):
            span_id = span_data['span_id']
            if span_id in visited:
                return # Avoid infinite loops in case of cycles
            visited.add(span_id)
            
            sorted_condensed_list.append(span_data) # Add parent before children

            # Get children, sort them by created_at, and visit them
            span_children = children_map.get(span_id, [])
            span_children.sort(key=lambda x: datetime.fromisoformat(x.get("created_at", "1970-01-01T00:00:00")))
            for child in span_children:
                # Ensure the child exists in our map before recursing
                if child['span_id'] in span_map: 
                    dfs(child)
                else:
                    # This case might indicate an issue, but we'll add the child directly
                    # if its parent was processed but the child itself wasn't in the initial list?
                    # Or if the child's 'enter' event was missing. For robustness, add it.
                    if child['span_id'] not in visited:
                         visited.add(child['span_id'])
                         sorted_condensed_list.append(child)


        # Start DFS from each root
        for root_span in roots:
            if root_span['span_id'] not in visited:
                dfs(root_span)
                
        # Handle spans that might not have been reachable from roots (orphans)
        # Though ideally, all spans should descend from a root.
        for span_data in spans_list:
             if span_data['span_id'] not in visited:
                  # Decide how to handle orphans, maybe append them at the end sorted by time?
                  # For now, let's just add them to ensure they aren't lost.
                  sorted_condensed_list.append(span_data)


        return sorted_condensed_list, evaluation_runs

    def save(self, overwrite: bool = False) -> Tuple[str, dict]:
        """
        Save the current trace to the database.
        Returns a tuple of (trace_id, trace_data) where trace_data is the trace data that was saved.
        """
        # Calculate total elapsed time
        total_duration = self.get_duration()
        
        raw_entries = [entry.to_dict() for entry in self.entries]
        
        condensed_entries, evaluation_runs = self.condense_trace(raw_entries)

        # Calculate total token counts from LLM API calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        total_prompt_tokens_cost = 0.0
        total_completion_tokens_cost = 0.0
        total_cost = 0.0
        
        # Only count tokens for actual LLM API call spans
        llm_span_names = {"OPENAI_API_CALL", "TOGETHER_API_CALL", "ANTHROPIC_API_CALL", "GOOGLE_API_CALL"}
        for entry in condensed_entries:
            if entry.get("span_type") == "llm" and entry.get("function") in llm_span_names and isinstance(entry.get("output"), dict):
                output = entry["output"]
                usage = output.get("usage", {})
                model_name = entry.get("inputs", {}).get("model", "")
                prompt_tokens = 0
                completion_tokens = 0   
                
                # Handle OpenAI/Together format
                if "prompt_tokens" in usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                # Handle Anthropic format
                elif "input_tokens" in usage:
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                
                total_tokens += usage.get("total_tokens", 0)
                
                # Calculate costs if model name is available
                if model_name:
                    try:
                        prompt_cost, completion_cost = cost_per_token(
                            model=model_name, 
                            prompt_tokens=prompt_tokens, 
                            completion_tokens=completion_tokens
                        )
                        total_prompt_tokens_cost += prompt_cost
                        total_completion_tokens_cost += completion_cost
                        total_cost += prompt_cost + completion_cost
                        
                        # Add cost information directly to the usage dictionary in the condensed entry
                        if "usage" not in output:
                            output["usage"] = {}
                        output["usage"]["prompt_tokens_cost_usd"] = prompt_cost
                        output["usage"]["completion_tokens_cost_usd"] = completion_cost
                        output["usage"]["total_cost_usd"] = prompt_cost + completion_cost
                    except Exception as e:
                        # If cost calculation fails, continue without adding costs
                        print(f"Error calculating cost for model '{model_name}': {str(e)}")
                        pass

        # Create trace document
        trace_data = {
            "trace_id": self.trace_id,
            "name": self.name,
            "project_name": self.project_name,
            "created_at": datetime.utcfromtimestamp(self.start_time).isoformat(),
            "duration": total_duration,
            "token_counts": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "prompt_tokens_cost_usd": total_prompt_tokens_cost,
                "completion_tokens_cost_usd": total_completion_tokens_cost,
                "total_cost_usd": total_cost
            },
            "entries": condensed_entries,
            "evaluation_runs": evaluation_runs,
            "overwrite": overwrite,
            "parent_trace_id": self.parent_trace_id,
            "parent_name": self.parent_name
        }        
        # --- Log trace data before saving ---
        try:
            rprint(f"[TraceClient.save] Saving trace data for trace_id {self.trace_id}:")
            rprint(json.dumps(trace_data, indent=2))
        except Exception as log_e:
            rprint(f"[TraceClient.save] Error logging trace data: {log_e}")
        # --- End logging ---
        self.trace_manager_client.save_trace(trace_data)

        return self.trace_id, trace_data

    def delete(self):
        return self.trace_manager_client.delete_trace(self.trace_id)
    
class Tracer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, 
        api_key: str = os.getenv("JUDGMENT_API_KEY"), 
        project_name: str = "default_project",
        rules: Optional[List[Rule]] = None,  # Added rules parameter
        organization_id: str = os.getenv("JUDGMENT_ORG_ID"),
        enable_monitoring: bool = os.getenv("JUDGMENT_MONITORING", "true").lower() == "true",
        enable_evaluations: bool = os.getenv("JUDGMENT_EVALUATIONS", "true").lower() == "true",
        # S3 configuration
        use_s3: bool = False,
        s3_bucket_name: Optional[str] = None,
        s3_aws_access_key_id: Optional[str] = None,
        s3_aws_secret_access_key: Optional[str] = None,
        s3_region_name: Optional[str] = None,
        deep_tracing: bool = True  # NEW: Enable deep tracing by default
        ):
        if not hasattr(self, 'initialized'):
            if not api_key:
                raise ValueError("Tracer must be configured with a Judgment API key")
            
            if not organization_id:
                raise ValueError("Tracer must be configured with an Organization ID")
            if use_s3 and not s3_bucket_name:
                raise ValueError("S3 bucket name must be provided when use_s3 is True")
            if use_s3 and not (s3_aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")):
                raise ValueError("AWS Access Key ID must be provided when use_s3 is True")
            if use_s3 and not (s3_aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")):
                raise ValueError("AWS Secret Access Key must be provided when use_s3 is True")
            
            self.api_key: str = api_key
            self.project_name: str = project_name
            self.client: JudgmentClient = JudgmentClient(judgment_api_key=api_key)
            self.organization_id: str = organization_id
            self._current_trace: Optional[str] = None
            self.rules: List[Rule] = rules or []  # Store rules at tracer level
            self.initialized: bool = True
            self.enable_monitoring: bool = enable_monitoring
            self.enable_evaluations: bool = enable_evaluations

            # Initialize S3 storage if enabled
            self.use_s3 = use_s3
            if use_s3:
                from judgeval.common.s3_storage import S3Storage
                self.s3_storage = S3Storage(
                    bucket_name=s3_bucket_name,
                    aws_access_key_id=s3_aws_access_key_id,
                    aws_secret_access_key=s3_aws_secret_access_key,
                    region_name=s3_region_name
                )
            self.deep_tracing: bool = deep_tracing  # NEW: Store deep tracing setting

        elif hasattr(self, 'project_name') and self.project_name != project_name:
            warnings.warn(
                f"Attempting to initialize Tracer with project_name='{project_name}' but it was already initialized with "
                f"project_name='{self.project_name}'. Due to the singleton pattern, the original project_name will be used. "
                "To use a different project name, ensure the first Tracer initialization uses the desired project name.",
                RuntimeWarning
            )
    
    def set_current_trace(self, trace: TraceClient):
        """
        Set the current trace context in contextvars
        """
        current_trace_var.set(trace)
    
    def get_current_trace(self) -> Optional[TraceClient]:
        """
        Get the current trace context from contextvars
        """
        return current_trace_var.get()
        
    def _apply_deep_tracing(self, func, span_type="span"):
        """
        Apply deep tracing to all functions in the same module as the given function.
        
        Args:
            func: The function being traced
            span_type: Type of span to use for traced functions
            
        Returns:
            A tuple of (module, original_functions_dict) where original_functions_dict
            contains the original functions that were replaced with traced versions.
        """
        module = inspect.getmodule(func)
        if not module:
            return None, {}
            
        # Save original functions
        original_functions = {}
        
        # Find all functions in the module
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # Skip already wrapped functions
            if hasattr(obj, '_judgment_traced'):
                continue
                
            # Create a traced version of the function
            # Always use default span type "span" for child functions
            traced_func = _create_deep_tracing_wrapper(obj, self, "span")
            
            # Mark the function as traced to avoid double wrapping
            traced_func._judgment_traced = True
            
            # Save the original function
            original_functions[name] = obj
            
            # Replace with traced version
            setattr(module, name, traced_func)
            
        return module, original_functions

    @contextmanager
    def trace(
        self, 
        name: str, 
        project_name: str = None, 
        overwrite: bool = False,
        rules: Optional[List[Rule]] = None  # Added rules parameter
    ) -> Generator[TraceClient, None, None]:
        """Start a new trace context using a context manager"""
        trace_id = str(uuid.uuid4())
        project = project_name if project_name is not None else self.project_name
        
        # Get parent trace info from context
        parent_trace = current_trace_var.get()
        parent_trace_id = None
        parent_name = None
        
        if parent_trace:
            parent_trace_id = parent_trace.trace_id
            parent_name = parent_trace.name

        trace = TraceClient(
            self, 
            trace_id, 
            name, 
            project_name=project, 
            overwrite=overwrite,
            rules=self.rules,  # Pass combined rules to the trace client
            enable_monitoring=self.enable_monitoring,
            enable_evaluations=self.enable_evaluations,
            parent_trace_id=parent_trace_id,
            parent_name=parent_name
        )
        
        # Set the current trace in context variables
        token = current_trace_var.set(trace)
        
        # Automatically create top-level span
        with trace.span(name or "unnamed_trace") as span:
            try:
                # Save the trace to the database to handle Evaluations' trace_id referential integrity
                yield trace
            finally:
                # Reset the context variable
                current_trace_var.reset(token)
    
    def observe(self, func=None, *, name=None, span_type: SpanType = "span", project_name: str = None, overwrite: bool = False, deep_tracing: bool = None):
        """
        Decorator to trace function execution with detailed entry/exit information.
        
        Args:
            func: The function to decorate
            name: Optional custom name for the span (defaults to function name)
            span_type: Type of span (default "span")
            project_name: Optional project name override
            overwrite: Whether to overwrite existing traces
            deep_tracing: Whether to enable deep tracing for this function and all nested calls.
                          If None, uses the tracer's default setting.
        """
        # If monitoring is disabled, return the function as is
        if not self.enable_monitoring:
            return func if func else lambda f: f
        
        if func is None:
            return lambda f: self.observe(f, name=name, span_type=span_type, project_name=project_name, 
                                         overwrite=overwrite, deep_tracing=deep_tracing)
        
        # Use provided name or fall back to function name
        span_name = name or func.__name__
        
        # Store custom attributes on the function object
        func._judgment_span_name = span_name
        func._judgment_span_type = span_type
        
        # Use the provided deep_tracing value or fall back to the tracer's default
        use_deep_tracing = deep_tracing if deep_tracing is not None else self.deep_tracing
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Check if we're already in a traced function
                if in_traced_function_var.get():
                    return await func(*args, **kwargs)
                
                # Set in_traced_function_var to True
                token = in_traced_function_var.set(True)
                
                # Get current trace from context
                current_trace = current_trace_var.get()
                
                # If there's no current trace, create a root trace
                if not current_trace:
                    trace_id = str(uuid.uuid4())
                    project = project_name if project_name is not None else self.project_name
                    
                    # Create a new trace client to serve as the root
                    current_trace = TraceClient(
                        self,
                        trace_id,
                        span_name, # MODIFIED: Use span_name directly
                        project_name=project,
                        overwrite=overwrite,
                        rules=self.rules,
                        enable_monitoring=self.enable_monitoring,
                        enable_evaluations=self.enable_evaluations
                    )
                    
                    # Save empty trace and set trace context
                    # current_trace.save(empty_save=True, overwrite=overwrite)
                    trace_token = current_trace_var.set(current_trace)
                    
                    try:
                        # Use span for the function execution within the root trace
                        # This sets the current_span_var
                        with current_trace.span(span_name, span_type=span_type) as span: # MODIFIED: Use span_name directly
                            # Record inputs
                            span.record_input({
                                'args': str(args),
                                'kwargs': kwargs
                            })
                            
                            # If deep tracing is enabled, apply monkey patching
                            if use_deep_tracing:
                                module, original_functions = self._apply_deep_tracing(func, span_type)
                            
                            # Execute function
                            result = await func(*args, **kwargs)
                            
                            # Restore original functions if deep tracing was enabled
                            if use_deep_tracing and module and 'original_functions' in locals():
                                for name, obj in original_functions.items():
                                    setattr(module, name, obj)
                            
                            # Record output
                            span.record_output(result)
                            
                        # Save the completed trace
                        current_trace.save(overwrite=overwrite)
                        return result
                    finally:
                        # Reset trace context (span context resets automatically)
                        current_trace_var.reset(trace_token)
                        # Reset in_traced_function_var
                        in_traced_function_var.reset(token)
                else:
                    # Already have a trace context, just create a span in it
                    # The span method handles current_span_var
                    
                    try:
                        with current_trace.span(span_name, span_type=span_type) as span: # MODIFIED: Use span_name directly
                            # Record inputs
                            span.record_input({
                                'args': str(args),
                                'kwargs': kwargs
                            })
                            
                            # If deep tracing is enabled, apply monkey patching
                            if use_deep_tracing:
                                module, original_functions = self._apply_deep_tracing(func, span_type)
                            
                            # Execute function
                            result = await func(*args, **kwargs)
                            
                            # Restore original functions if deep tracing was enabled
                            if use_deep_tracing and module and 'original_functions' in locals():
                                for name, obj in original_functions.items():
                                    setattr(module, name, obj)
                            
                            # Record output
                            span.record_output(result)
                        
                        return result
                    finally:
                        # Reset in_traced_function_var
                        in_traced_function_var.reset(token)
                
            return async_wrapper
        else:
            # Non-async function implementation with deep tracing
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Check if we're already in a traced function
                if in_traced_function_var.get():
                    return func(*args, **kwargs)
                
                # Set in_traced_function_var to True
                token = in_traced_function_var.set(True)
                
                # Get current trace from context
                current_trace = current_trace_var.get()
                
                # If there's no current trace, create a root trace
                if not current_trace:
                    trace_id = str(uuid.uuid4())
                    project = project_name if project_name is not None else self.project_name
                    
                    # Create a new trace client to serve as the root
                    current_trace = TraceClient(
                        self,
                        trace_id,
                        span_name, # MODIFIED: Use span_name directly
                        project_name=project,
                        overwrite=overwrite,
                        rules=self.rules,
                        enable_monitoring=self.enable_monitoring,
                        enable_evaluations=self.enable_evaluations
                    )
                    
                    # Save empty trace and set trace context
                    # current_trace.save(empty_save=True, overwrite=overwrite)
                    trace_token = current_trace_var.set(current_trace)
                    
                    try:
                        # Use span for the function execution within the root trace
                        # This sets the current_span_var
                        with current_trace.span(span_name, span_type=span_type) as span: # MODIFIED: Use span_name directly
                            # Record inputs
                            span.record_input({
                                'args': str(args),
                                'kwargs': kwargs
                            })
                            
                            # If deep tracing is enabled, apply monkey patching
                            if use_deep_tracing:
                                module, original_functions = self._apply_deep_tracing(func, span_type)
                            
                            # Execute function
                            result = func(*args, **kwargs)
                            
                            # Restore original functions if deep tracing was enabled
                            if use_deep_tracing and module and 'original_functions' in locals():
                                for name, obj in original_functions.items():
                                    setattr(module, name, obj)
                            
                            # Record output
                            span.record_output(result)
                            
                        # Save the completed trace
                        current_trace.save(overwrite=overwrite)
                        return result
                    finally:
                        # Reset trace context (span context resets automatically)
                        current_trace_var.reset(trace_token)
                        # Reset in_traced_function_var
                        in_traced_function_var.reset(token)
                else:
                    # Already have a trace context, just create a span in it
                    # The span method handles current_span_var
                    
                    try:
                        with current_trace.span(span_name, span_type=span_type) as span: # MODIFIED: Use span_name directly
                            # Record inputs
                            span.record_input({
                                'args': str(args),
                                'kwargs': kwargs
                            })
                            
                            # If deep tracing is enabled, apply monkey patching
                            if use_deep_tracing:
                                module, original_functions = self._apply_deep_tracing(func, span_type)
                            
                            # Execute function
                            result = func(*args, **kwargs)
                            
                            # Restore original functions if deep tracing was enabled
                            if use_deep_tracing and module and 'original_functions' in locals():
                                for name, obj in original_functions.items():
                                    setattr(module, name, obj)
                            
                            # Record output
                            span.record_output(result)
                        
                        return result
                    finally:
                        # Reset in_traced_function_var
                        in_traced_function_var.reset(token)
                
            return wrapper
        
    def score(self, func=None, scorers: List[Union[APIJudgmentScorer, JudgevalScorer]] = None, model: str = None, log_results: bool = True, *, name: str = None, span_type: SpanType = "span"):
        """
        Decorator to trace function execution with detailed entry/exit information.
        """
        if func is None:
            return lambda f: self.score(f, scorers=scorers, model=model, log_results=log_results, name=name, span_type=span_type)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get current trace from contextvars
                current_trace = current_trace_var.get()
                if current_trace and scorers:
                    current_trace.async_evaluate(scorers=scorers, input=args, actual_output=kwargs, model=model, log_results=log_results)
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Get current trace from contextvars
                current_trace = current_trace_var.get()
                if current_trace and scorers:
                    current_trace.async_evaluate(scorers=scorers, input=args, actual_output=kwargs, model=model, log_results=log_results)
                return func(*args, **kwargs)
            return wrapper
        
    def async_evaluate(self, *args, **kwargs):
        if not self.enable_evaluations:
            return

        # Get current trace from context
        current_trace = current_trace_var.get()
        
        if current_trace:
            current_trace.async_evaluate(*args, **kwargs)
        else:
            warnings.warn("No trace found, skipping evaluation")


def wrap(client: Any) -> Any:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, and Google GenAI clients.
    Patches both '.create' and Anthropic's '.stream' methods using a wrapper class.
    """
    span_name, original_create, original_stream = _get_client_config(client)

    # --- Define Traced Async Functions ---
    async def traced_create_async(*args, **kwargs):
        # [Existing logic - unchanged]
        current_trace = current_trace_var.get()
        if not current_trace:
            if asyncio.iscoroutinefunction(original_create):
                 return await original_create(*args, **kwargs)
            else:
                 return original_create(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)

        with current_trace.span(span_name, span_type="llm") as span:
            input_data = _format_input_data(client, **kwargs)
            span.record_input(input_data)

            # Warn about token counting limitations with streaming
            if isinstance(client, (AsyncOpenAI, OpenAI)) and is_streaming:
                if not kwargs.get("stream_options", {}).get("include_usage"):
                    warnings.warn(
                        "OpenAI streaming calls don't include token counts by default. "
                        "To enable token counting with streams, set stream_options={'include_usage': True} "
                        "in your API call arguments.",
                        UserWarning
                    )

            try:
                if is_streaming:
                    stream_iterator = await original_create(*args, **kwargs)
                    output_entry = span.record_output("<pending stream>")
                    return _async_stream_wrapper(stream_iterator, client, output_entry)
                else:
                    awaited_response = await original_create(*args, **kwargs)
                    output_data = _format_output_data(client, awaited_response)
                    span.record_output(output_data)
                    return awaited_response
            except Exception as e:
                print(f"Error during wrapped async API call ({span_name}): {e}")
                span.record_output({"error": str(e)})
                raise


    # Function replacing .stream() - NOW returns the wrapper class instance
    def traced_stream_async(*args, **kwargs):
        current_trace = current_trace_var.get()
        if not current_trace or not original_stream:
            return original_stream(*args, **kwargs)
        original_manager = original_stream(*args, **kwargs)
        wrapper_manager = _TracedAsyncStreamManagerWrapper(
            original_manager=original_manager,
            client=client,
            span_name=span_name,
            trace_client=current_trace,
            stream_wrapper_func=_async_stream_wrapper,
            input_kwargs=kwargs
        )
        return wrapper_manager

    # --- Define Traced Sync Functions ---
    def traced_create_sync(*args, **kwargs):
         # [Existing logic - unchanged]
        current_trace = current_trace_var.get()
        if not current_trace:
             return original_create(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)

        with current_trace.span(span_name, span_type="llm") as span:
             input_data = _format_input_data(client, **kwargs)
             span.record_input(input_data)

             # Warn about token counting limitations with streaming
             if isinstance(client, (AsyncOpenAI, OpenAI)) and is_streaming:
                 if not kwargs.get("stream_options", {}).get("include_usage"):
                     warnings.warn(
                         "OpenAI streaming calls don't include token counts by default. "
                         "To enable token counting with streams, set stream_options={'include_usage': True} "
                         "in your API call arguments.",
                         UserWarning
                     )

             try:
                 response_or_iterator = original_create(*args, **kwargs)
             except Exception as e:
                 print(f"Error during wrapped sync API call ({span_name}): {e}")
                 span.record_output({"error": str(e)})
                 raise

             if is_streaming:
                 output_entry = span.record_output("<pending stream>")
                 return _sync_stream_wrapper(response_or_iterator, client, output_entry)
             else:
                 output_data = _format_output_data(client, response_or_iterator)
                 span.record_output(output_data)
                 return response_or_iterator


    # Function replacing sync .stream()
    def traced_stream_sync(*args, **kwargs):
         current_trace = current_trace_var.get()
         if not current_trace or not original_stream:
             return original_stream(*args, **kwargs)
         original_manager = original_stream(*args, **kwargs)
         wrapper_manager = _TracedSyncStreamManagerWrapper(
             original_manager=original_manager,
             client=client,
             span_name=span_name,
             trace_client=current_trace,
             stream_wrapper_func=_sync_stream_wrapper,
             input_kwargs=kwargs
         )
         return wrapper_manager


    # --- Assign Traced Methods to Client Instance ---
    # [Assignment logic remains the same]
    if isinstance(client, (AsyncOpenAI, AsyncTogether)):
        client.chat.completions.create = traced_create_async
        # Wrap the Responses API endpoint for OpenAI clients
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            # Capture the original responses.create
            original_responses_create = client.responses.create
            def traced_responses(*args, **kwargs):
                # Get the current trace from contextvars
                current_trace = current_trace_var.get()
                # If no active trace, call the original
                if not current_trace:
                    return original_responses_create(*args, **kwargs)
                # Trace this responses.create call
                with current_trace.span(span_name, span_type="llm") as span:
                    # Record raw input kwargs
                    span.record_input(kwargs)
                    # Make the actual API call
                    response = original_responses_create(*args, **kwargs)
                    # Record the output object
                    span.record_output(response)
                    return response
            # Assign the traced wrapper
            client.responses.create = traced_responses
    elif isinstance(client, AsyncAnthropic):
        client.messages.create = traced_create_async
        if original_stream:
             client.messages.stream = traced_stream_async
    elif isinstance(client, genai.client.AsyncClient):
        client.generate_content = traced_create_async
    elif isinstance(client, (OpenAI, Together)):
         client.chat.completions.create = traced_create_sync
    elif isinstance(client, Anthropic):
         client.messages.create = traced_create_sync
         if original_stream:
             client.messages.stream = traced_stream_sync
    elif isinstance(client, genai.Client):
         client.generate_content = traced_create_sync

    return client

# Helper functions for client-specific operations

def _get_client_config(client: ApiClient) -> tuple[str, callable, Optional[callable]]:
    """Returns configuration tuple for the given API client.
    
    Args:
        client: An instance of OpenAI, Together, or Anthropic client
        
    Returns:
        tuple: (span_name, create_method, stream_method)
            - span_name: String identifier for tracing
            - create_method: Reference to the client's creation method
            - stream_method: Reference to the client's stream method (if applicable)
            
    Raises:
        ValueError: If client type is not supported
    """
    if isinstance(client, (OpenAI, AsyncOpenAI)):
        return "OPENAI_API_CALL", client.chat.completions.create, None
    elif isinstance(client, (Together, AsyncTogether)):
        return "TOGETHER_API_CALL", client.chat.completions.create, None
    elif isinstance(client, (Anthropic, AsyncAnthropic)):
        return "ANTHROPIC_API_CALL", client.messages.create, client.messages.stream
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        return "GOOGLE_API_CALL", client.models.generate_content, None
    raise ValueError(f"Unsupported client type: {type(client)}")

def _format_input_data(client: ApiClient, **kwargs) -> dict:
    """Format input parameters based on client type.
    
    Extracts relevant parameters from kwargs based on the client type
    to ensure consistent tracing across different APIs.
    """
    if isinstance(client, (OpenAI, Together, AsyncOpenAI, AsyncTogether)):
        return {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
        }
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        return {
            "model": kwargs.get("model"),
            "contents": kwargs.get("contents")
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
    if isinstance(client, (OpenAI, Together, AsyncOpenAI, AsyncTogether)):
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        return {
            "content": response.candidates[0].content.parts[0].text,
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
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

# Define a blocklist of functions that should not be traced
# These are typically utility functions, print statements, logging, etc.
_TRACE_BLOCKLIST = {
    # Built-in functions
    'print', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
    'min', 'max', 'sum', 'any', 'all', 'abs', 'round', 'format',
    # Logging functions
    'debug', 'info', 'warning', 'error', 'critical', 'exception', 'log',
    # Common utility functions
    'sleep', 'time', 'datetime', 'json', 'dumps', 'loads',
    # String operations
    'join', 'split', 'strip', 'lstrip', 'rstrip', 'replace', 'lower', 'upper',
    # Dict operations
    'get', 'items', 'keys', 'values', 'update',
    # List operations
    'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'index', 'count', 'sort',
}


# Add a new function for deep tracing at the module level
def _create_deep_tracing_wrapper(func, tracer, span_type="span"):
    """
    Creates a wrapper for a function that automatically traces it when called within a traced function.
    This enables deep tracing without requiring explicit @observe decorators on every function.
    
    Args:
        func: The function to wrap
        tracer: The Tracer instance
        span_type: Type of span (default "span")
        
    Returns:
        A wrapped function that will be traced when called
    """
    # Skip wrapping if the function is not callable or is a built-in
    if not callable(func) or isinstance(func, type) or func.__module__ == 'builtins':
        return func
    
    # Skip functions in the blocklist
    if func.__name__ in _TRACE_BLOCKLIST:
        return func
    
    # Skip functions from certain modules (logging, sys, etc.)
    if func.__module__ and any(func.__module__.startswith(m) for m in ['logging', 'sys', 'os', 'json', 'time', 'datetime']):
        return func
    

    # Get function name for the span - check for custom name set by @observe
    func_name = getattr(func, '_judgment_span_name', func.__name__)
    
    # Check for custom span_type set by @observe
    func_span_type = getattr(func, '_judgment_span_type', "span")
    
    # Store original function to prevent losing reference
    original_func = func
    
    # Create appropriate wrapper based on whether the function is async or not
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_deep_wrapper(*args, **kwargs):
            # Get current trace from context
            current_trace = current_trace_var.get()
            
            # If no trace context, just call the function
            if not current_trace:
                return await original_func(*args, **kwargs)
            
            # Create a span for this function call - use custom span_type if available
            with current_trace.span(func_name, span_type=func_span_type) as span:
                # Record inputs
                span.record_input({
                    'args': str(args),
                    'kwargs': kwargs
                })
                
                # Execute function
                result = await original_func(*args, **kwargs)
                
                # Record output
                span.record_output(result)
                
                return result
                
        return async_deep_wrapper
    else:
        @functools.wraps(func)
        def deep_wrapper(*args, **kwargs):
            # Get current trace from context
            current_trace = current_trace_var.get()
            
            # If no trace context, just call the function
            if not current_trace:
                return original_func(*args, **kwargs)
            
            # Create a span for this function call - use custom span_type if available
            with current_trace.span(func_name, span_type=func_span_type) as span:
                # Record inputs
                span.record_input({
                    'args': str(args),
                    'kwargs': kwargs
                })
                
                # Execute function
                result = original_func(*args, **kwargs)
                
                # Record output
                span.record_output(result)
                
                return result
                
        return deep_wrapper

# Add the new TraceThreadPoolExecutor class
class TraceThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """
    A ThreadPoolExecutor subclass that automatically propagates contextvars
    from the submitting thread to the worker thread using copy_context().run().

    This ensures that context variables like `current_trace_var` and
    `current_span_var` are available within functions executed by the pool,
    allowing the Tracer to maintain correct parent-child relationships across
    thread boundaries.
    """
    def submit(self, fn, /, *args, **kwargs):
        """
        Submit a callable to be executed with the captured context.
        """
        # Capture context from the submitting thread
        ctx = contextvars.copy_context()

        # We use functools.partial to bind the arguments to the function *now*,
        # as ctx.run doesn't directly accept *args, **kwargs in the same way
        # submit does. It expects ctx.run(callable, arg1, arg2...).
        func_with_bound_args = functools.partial(fn, *args, **kwargs)

        # Submit the ctx.run callable to the original executor.
        # ctx.run will execute the (now argument-bound) function within the
        # captured context in the worker thread.
        return super().submit(ctx.run, func_with_bound_args)

    # Note: The `map` method would also need to be overridden for full context
    # propagation if users rely on it, but `submit` is the most common use case.

# Helper functions for stream processing
# ---------------------------------------

def _extract_content_from_chunk(client: ApiClient, chunk: Any) -> Optional[str]:
    """Extracts the text content from a stream chunk based on the client type."""
    try:
        if isinstance(client, (OpenAI, Together, AsyncOpenAI, AsyncTogether)):
            return chunk.choices[0].delta.content
        elif isinstance(client, (Anthropic, AsyncAnthropic)):
            # Anthropic streams various event types, we only care for content blocks
            if chunk.type == "content_block_delta":
                return chunk.delta.text
        elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
            # Google streams Candidate objects
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                return chunk.candidates[0].content.parts[0].text
    except (AttributeError, IndexError, KeyError):
        # Handle cases where chunk structure is unexpected or doesn't contain content
        pass # Return None
    return None

def _extract_usage_from_final_chunk(client: ApiClient, chunk: Any) -> Optional[Dict[str, int]]:
    """Extracts usage data if present in the *final* chunk (client-specific)."""
    try:
        # OpenAI/Together include usage in the *last* chunk's `usage` attribute if available
        # This typically requires specific API versions or settings. Often usage is *not* streamed.
        if isinstance(client, (OpenAI, Together, AsyncOpenAI, AsyncTogether)):
             # Check if usage is directly on the chunk (some models might do this)
             if hasattr(chunk, 'usage') and chunk.usage:
                 return {
                     "prompt_tokens": chunk.usage.prompt_tokens,
                     "completion_tokens": chunk.usage.completion_tokens,
                     "total_tokens": chunk.usage.total_tokens
                 }
             # Check if usage is nested within choices (less common for final chunk, but check)
             elif chunk.choices and hasattr(chunk.choices[0], 'usage') and chunk.choices[0].usage:
                 usage = chunk.choices[0].usage
                 return {
                      "prompt_tokens": usage.prompt_tokens,
                      "completion_tokens": usage.completion_tokens,
                      "total_tokens": usage.total_tokens
                  }
             # Anthropic includes usage in the 'message_stop' event type
        elif isinstance(client, (Anthropic, AsyncAnthropic)):
            if chunk.type == "message_stop":
                # Anthropic final usage is often attached to the *message* object, not the chunk directly
                # The API might provide a way to get the final message object, but typically not in the stream itself.
                # Let's assume for now usage might appear in the final *chunk* metadata if supported.
                # This is a placeholder - Anthropic usage typically needs a separate call or context.
                pass
        elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
             # Google provides usage metadata on the full response object, not typically streamed per chunk.
             # It might be in the *last* chunk's usage_metadata if the stream implementation supports it.
             if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                 return {
                     "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                     "completion_tokens": chunk.usage_metadata.candidates_token_count,
                     "total_tokens": chunk.usage_metadata.total_token_count
                 }

    except (AttributeError, IndexError, KeyError, TypeError):
        # Handle cases where usage data is missing or malformed
         pass # Return None
    return None


# --- Sync Stream Wrapper ---
def _sync_stream_wrapper(
    original_stream: Iterator,
    client: ApiClient,
    output_entry: TraceEntry
) -> Generator[Any, None, None]:
    """Wraps a synchronous stream iterator to capture content and update the trace."""
    content_parts = []  # Use a list instead of string concatenation
    final_usage = None
    last_chunk = None
    try:
        for chunk in original_stream:
            content_part = _extract_content_from_chunk(client, chunk)
            if content_part:
                content_parts.append(content_part)  # Append to list instead of concatenating
            last_chunk = chunk # Keep track of the last chunk for potential usage data
            yield chunk # Pass the chunk to the caller
    finally:
        # Attempt to extract usage from the last chunk received
        if last_chunk:
            final_usage = _extract_usage_from_final_chunk(client, last_chunk)

        # Update the trace entry with the accumulated content and usage
        output_entry.output = {
            "content": "".join(content_parts),  # Join list at the end
            "usage": final_usage if final_usage else {"info": "Usage data not available in stream."}, # Provide placeholder if None
            "streamed": True
        }
        # Note: We might need to adjust _serialize_output if this dict causes issues,
        # but Pydantic's model_dump should handle dicts.

# --- Async Stream Wrapper ---
async def _async_stream_wrapper(
    original_stream: AsyncIterator,
    client: ApiClient,
    output_entry: TraceEntry
) -> AsyncGenerator[Any, None]:
    # [Existing logic - unchanged]
    content_parts = []  # Use a list instead of string concatenation
    final_usage_data = None
    last_content_chunk = None
    anthropic_input_tokens = 0
    anthropic_output_tokens = 0

    target_span_id = getattr(output_entry, 'span_id', 'UNKNOWN')

    try:
        async for chunk in original_stream:
            # Check for OpenAI's final usage chunk
            if isinstance(client, (AsyncOpenAI, OpenAI)) and hasattr(chunk, 'usage') and chunk.usage is not None:
                final_usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens
                }
                yield chunk
                continue

            if isinstance(client, (AsyncAnthropic, Anthropic)) and hasattr(chunk, 'type'):
                 if chunk.type == "message_start":
                     if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage') and hasattr(chunk.message.usage, 'input_tokens'):
                         anthropic_input_tokens = chunk.message.usage.input_tokens
                 elif chunk.type == "message_delta":
                     if hasattr(chunk, 'usage') and hasattr(chunk.usage, 'output_tokens'):
                         anthropic_output_tokens += chunk.usage.output_tokens

            content_part = _extract_content_from_chunk(client, chunk)
            if content_part:
                content_parts.append(content_part)  # Append to list instead of concatenating
                last_content_chunk = chunk

            yield chunk
    finally:
        anthropic_final_usage = None
        if isinstance(client, (AsyncAnthropic, Anthropic)) and (anthropic_input_tokens > 0 or anthropic_output_tokens > 0):
             anthropic_final_usage = {
                 "input_tokens": anthropic_input_tokens,
                 "output_tokens": anthropic_output_tokens,
                 "total_tokens": anthropic_input_tokens + anthropic_output_tokens
             }

        usage_info = None
        if final_usage_data:
             usage_info = final_usage_data
        elif anthropic_final_usage:
             usage_info = anthropic_final_usage
        elif last_content_chunk:
             usage_info = _extract_usage_from_final_chunk(client, last_content_chunk)

        if output_entry and hasattr(output_entry, 'output'):
            output_entry.output = {
                "content": "".join(content_parts),  # Join list at the end
                "usage": usage_info if usage_info else {"info": "Usage data not available in stream."},
                "streamed": True
            }
            start_ts = getattr(output_entry, 'created_at', time.time())
            output_entry.duration = time.time() - start_ts
        # else: # Handle error case if necessary, but remove debug print

# --- Define Context Manager Wrapper Classes ---
class _TracedAsyncStreamManagerWrapper(AbstractAsyncContextManager):
    """Wraps an original async stream manager to add tracing."""
    def __init__(self, original_manager, client, span_name, trace_client, stream_wrapper_func, input_kwargs):
        self._original_manager = original_manager
        self._client = client
        self._span_name = span_name
        self._trace_client = trace_client
        self._stream_wrapper_func = stream_wrapper_func
        self._input_kwargs = input_kwargs
        self._parent_span_id_at_entry = None

    async def __aenter__(self):
        self._parent_span_id_at_entry = current_span_var.get()
        if not self._trace_client:
             # If no trace, just delegate to the original manager
             return await self._original_manager.__aenter__()

        # --- Manually create the 'enter' entry ---
        start_time = time.time()
        span_id = str(uuid.uuid4())
        current_depth = 0
        if self._parent_span_id_at_entry and self._parent_span_id_at_entry in self._trace_client._span_depths:
            current_depth = self._trace_client._span_depths[self._parent_span_id_at_entry] + 1
        self._trace_client._span_depths[span_id] = current_depth
        enter_entry = TraceEntry(
             type="enter", function=self._span_name, span_id=span_id,
             trace_id=self._trace_client.trace_id, depth=current_depth, message=self._span_name,
             created_at=start_time, span_type="llm", parent_span_id=self._parent_span_id_at_entry
        )
        self._trace_client.add_entry(enter_entry)
        # --- End manual 'enter' entry ---

        # Set the current span ID in contextvars
        self._span_context_token = current_span_var.set(span_id)

        # Manually create 'input' entry
        input_data = _format_input_data(self._client, **self._input_kwargs)
        input_entry = TraceEntry(
             type="input", function=self._span_name, span_id=span_id,
             trace_id=self._trace_client.trace_id, depth=current_depth, message=f"Inputs to {self._span_name}",
             created_at=time.time(), inputs=input_data, span_type="llm"
        )
        self._trace_client.add_entry(input_entry)

        # Call the original __aenter__
        raw_iterator = await self._original_manager.__aenter__()

        # Manually create pending 'output' entry
        output_entry = TraceEntry(
            type="output", function=self._span_name, span_id=span_id,
            trace_id=self._trace_client.trace_id, depth=current_depth, message=f"Output from {self._span_name}",
            created_at=time.time(), output="<pending stream>", span_type="llm"
        )
        self._trace_client.add_entry(output_entry)

        # Wrap the raw iterator
        wrapped_iterator = self._stream_wrapper_func(raw_iterator, self._client, output_entry)
        return wrapped_iterator

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Manually create the 'exit' entry
        if hasattr(self, '_span_context_token'):
             span_id = current_span_var.get()
             start_time_for_duration = 0
             for entry in reversed(self._trace_client.entries):
                  if entry.span_id == span_id and entry.type == 'enter':
                       start_time_for_duration = entry.created_at
                       break
             duration = time.time() - start_time_for_duration if start_time_for_duration else None
             exit_depth = self._trace_client._span_depths.get(span_id, 0)
             exit_entry = TraceEntry(
                  type="exit", function=self._span_name, span_id=span_id,
                  trace_id=self._trace_client.trace_id, depth=exit_depth, message=f"← {self._span_name}",
                  created_at=time.time(), duration=duration, span_type="llm"
             )
             self._trace_client.add_entry(exit_entry)
             if span_id in self._trace_client._span_depths: del self._trace_client._span_depths[span_id]
             current_span_var.reset(self._span_context_token)
             delattr(self, '_span_context_token')

        # Delegate __aexit__
        if hasattr(self._original_manager, "__aexit__"):
             return await self._original_manager.__aexit__(exc_type, exc_val, exc_tb)
        return None

class _TracedSyncStreamManagerWrapper(AbstractContextManager):
    """Wraps an original sync stream manager to add tracing."""
    def __init__(self, original_manager, client, span_name, trace_client, stream_wrapper_func, input_kwargs):
        self._original_manager = original_manager
        self._client = client
        self._span_name = span_name
        self._trace_client = trace_client
        self._stream_wrapper_func = stream_wrapper_func
        self._input_kwargs = input_kwargs
        self._parent_span_id_at_entry = None

    def __enter__(self):
        self._parent_span_id_at_entry = current_span_var.get()
        if not self._trace_client:
             return self._original_manager.__enter__()

        # Manually create 'enter' entry
        start_time = time.time()
        span_id = str(uuid.uuid4())
        current_depth = 0
        if self._parent_span_id_at_entry and self._parent_span_id_at_entry in self._trace_client._span_depths:
            current_depth = self._trace_client._span_depths[self._parent_span_id_at_entry] + 1
        self._trace_client._span_depths[span_id] = current_depth
        enter_entry = TraceEntry(
             type="enter", function=self._span_name, span_id=span_id,
             trace_id=self._trace_client.trace_id, depth=current_depth, message=self._span_name,
             created_at=start_time, span_type="llm", parent_span_id=self._parent_span_id_at_entry
        )
        self._trace_client.add_entry(enter_entry)
        self._span_context_token = current_span_var.set(span_id)

        # Manually create 'input' entry
        input_data = _format_input_data(self._client, **self._input_kwargs)
        input_entry = TraceEntry(
             type="input", function=self._span_name, span_id=span_id,
             trace_id=self._trace_client.trace_id, depth=current_depth, message=f"Inputs to {self._span_name}",
             created_at=time.time(), inputs=input_data, span_type="llm"
        )
        self._trace_client.add_entry(input_entry)

        # Call original __enter__
        raw_iterator = self._original_manager.__enter__()

        # Manually create 'output' entry (pending)
        output_entry = TraceEntry(
            type="output", function=self._span_name, span_id=span_id,
            trace_id=self._trace_client.trace_id, depth=current_depth, message=f"Output from {self._span_name}",
            created_at=time.time(), output="<pending stream>", span_type="llm"
        )
        self._trace_client.add_entry(output_entry)

        # Wrap the raw iterator
        wrapped_iterator = self._stream_wrapper_func(raw_iterator, self._client, output_entry)
        return wrapped_iterator

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Manually create 'exit' entry
        if hasattr(self, '_span_context_token'):
             span_id = current_span_var.get()
             start_time_for_duration = 0
             for entry in reversed(self._trace_client.entries):
                  if entry.span_id == span_id and entry.type == 'enter':
                       start_time_for_duration = entry.created_at
                       break
             duration = time.time() - start_time_for_duration if start_time_for_duration else None
             exit_depth = self._trace_client._span_depths.get(span_id, 0)
             exit_entry = TraceEntry(
                  type="exit", function=self._span_name, span_id=span_id,
                  trace_id=self._trace_client.trace_id, depth=exit_depth, message=f"← {self._span_name}",
                  created_at=time.time(), duration=duration, span_type="llm"
             )
             self._trace_client.add_entry(exit_entry)
             if span_id in self._trace_client._span_depths: del self._trace_client._span_depths[span_id]
             current_span_var.reset(self._span_context_token)
             delattr(self, '_span_context_token')

        # Delegate __exit__
        if hasattr(self._original_manager, "__exit__"):
             return self._original_manager.__exit__(exc_type, exc_val, exc_tb)
        return None
