"""
Tracing system for judgeval that allows for function tracing using decorators.
"""

import os
import time
import functools
import requests
import uuid
from contextlib import contextmanager
from typing import (
    Optional, 
    Any, 
    List, 
    Literal, 
    Tuple, 
    Generator, 
    TypeAlias, 
    Union
)
from dataclasses import (
    dataclass, 
    field
)
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
from rich import print as rprint
from uuid import UUID
from collections.abc import Sequence

import pika
import os

from judgeval.constants import JUDGMENT_TRACES_SAVE_API_URL, JUDGMENT_TRACES_FETCH_API_URL, RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_QUEUE, JUDGMENT_TRACES_DELETE_API_URL,JUDGMENT_TRACES_ADD_TO_EVAL_QUEUE_API_URL
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import APIJudgmentScorer, JudgevalScorer, ScorerWrapper
from judgeval.rules import Rule
from judgeval.evaluation_run import EvaluationRun
from judgeval.judges import JudgevalJudge

from rich import print as rprint

from judgeval.data.result import ScoringResult
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.documents import Document


# Define type aliases for better code readability and maintainability
ApiClient: TypeAlias = Union[OpenAI, Together, Anthropic]  # Supported API clients
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
    function: str  # Name of the function being traced
    depth: int    # Indentation level for nested calls
    message: str  # Human-readable description
    timestamp: float  # Unix timestamp when entry was created
    duration: Optional[float] = None  # Time taken (for exit/evaluation entries)
    output: Any = None  # Function output value
    # Use field() for mutable defaults to avoid shared state issues
    inputs: dict = field(default_factory=dict)
    span_type: SpanType = "span"
    evaluation_runs: List[Optional[EvaluationRun]] = field(default=None)
    
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
            for evaluation_run in self.evaluation_runs:
                print(f"{indent}Evaluation: {evaluation_run.model_dump()}")
    
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
            "evaluation_runs": [evaluation_run.model_dump() for evaluation_run in self.evaluation_runs] if self.evaluation_runs else [],
            "span_type": self.span_type
        }

    def _serialize_output(self) -> Any:
        """Helper method to serialize output data safely.
        
        Handles special cases:
        - Pydantic models are converted using model_dump()
        - We try to serialize into JSON, then string, then the base representation (__repr__)
        - Non-serializable objects return None with a warning
        """

        def safe_stringify(output, function_name):
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
        
        if isinstance(self.output, BaseModel):
            return self.output.model_dump()
        
        try:
            # Try to serialize the output to verify it's JSON compatible
            json.dumps(self.output)
            return self.output
        except (TypeError, OverflowError, ValueError):
            return safe_stringify(self.output, self.function)
        

class TraceManagerClient:
    """
    Client for handling trace endpoints with the Judgment API
    

    Operations include:
    - Fetching a trace by id
    - Saving a trace
    - Deleting a trace
    """
    def __init__(self, judgment_api_key: str, organization_id: str):
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id

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

    def save_trace(self, trace_data: dict, empty_save: bool):
        """
        Saves a trace to the database

        Args:
            trace_data: The trace data to save
            empty_save: Whether to save an empty trace
            NOTE we save empty traces in order to properly handle async operations; we need something in the DB to associate the async results with
        """
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
        if not empty_save and "ui_results_url" in response.json():
            rprint(f"\n🔍 You can view your trace data here: [rgb(106,0,255)]{response.json()['ui_results_url']}[/]\n")

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
    ):
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.project_name = project_name
        self.overwrite = overwrite
        self.tracer = tracer
        # Initialize rules with either provided rules or an empty list
        self.rules = rules or []
        
        self.client: JudgmentClient = tracer.client
        self.entries: List[TraceEntry] = []
        self.start_time = time.time()
        self.span_type = None
        self._current_span: Optional[TraceEntry] = None
        self.trace_manager_client = TraceManagerClient(tracer.api_key, tracer.organization_id)  # Manages DB operations for trace data
        
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
        
        # Increment nested depth and set current span
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
        loaded_rules = None
        if self.rules:
            loaded_rules = []
            for rule in self.rules:
                processed_conditions = []
                for condition in rule.conditions:
                    # Convert metric if it's a ScorerWrapper
                    try:
                        if isinstance(condition.metric, ScorerWrapper):
                            condition_copy = condition.model_copy()
                            condition_copy.metric = condition.metric.load_implementation(use_judgment=True)
                            processed_conditions.append(condition_copy)
                        else:
                            processed_conditions.append(condition)
                    except Exception as e:
                        warnings.warn(f"Failed to convert ScorerWrapper in rule '{rule.name}', condition metric '{condition.metric_name}': {str(e)}")
                        processed_conditions.append(condition)  # Keep original condition as fallback
                
                # Create new rule with processed conditions
                new_rule = rule.model_copy()
                new_rule.conditions = processed_conditions
                loaded_rules.append(new_rule)
        try:
            # Load appropriate implementations for all scorers
            loaded_scorers: List[Union[JudgevalScorer, APIJudgmentScorer]] = []
            for scorer in scorers:
                try:
                    if isinstance(scorer, ScorerWrapper):
                        loaded_scorers.append(scorer.load_implementation(use_judgment=True))
                    else:
                        loaded_scorers.append(scorer)
                except Exception as e:
                    warnings.warn(f"Failed to load implementation for scorer {scorer}: {str(e)}")
                    # Skip this scorer
            
            if not loaded_scorers:
                warnings.warn("No valid scorers available for evaluation")
                return
            
            # Prevent using JudgevalScorer with rules - only APIJudgmentScorer allowed with rules
            if loaded_rules and any(isinstance(scorer, JudgevalScorer) for scorer in loaded_scorers):
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
                f"{self._current_span}-"
                f"[{','.join(scorer.score_type.capitalize() for scorer in loaded_scorers)}]",
            examples=[example],
            scorers=loaded_scorers,
            model=model,
            metadata={},
            judgment_api_key=self.tracer.api_key,
            override=self.overwrite,
            rules=loaded_rules # Use the combined rules
        )
        
        self.add_eval_run(eval_run, start_time)  # Pass start_time to record_evaluation
            
    def add_eval_run(self, eval_run: EvaluationRun, start_time: float):
        """
        Add evaluation run data to the trace

        Args:
            eval_run (EvaluationRun): The evaluation run to add to the trace
            start_time (float): The start time of the evaluation run
        """
        if self._current_span:
            duration = time.time() - start_time  # Calculate duration from start_time
            
            prev_entry = self.entries[-1]
            
            # Select the last entry in the trace if it's an LLM call, otherwise use the current span
            self.add_entry(TraceEntry(
                type="evaluation",
                function=prev_entry.function if prev_entry.span_type == "llm" else self._current_span,
                depth=self.tracer.depth,
                message=f"Evaluation results for {self._current_span}",
                timestamp=time.time(),
                evaluation_runs=[eval_run],
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

        for i, entry in enumerate(entries):
            function = entry["function"]
            
            if entry["type"] == "enter":
                # Initialize new function entry
                function_entries[function] = {
                    "depth": entry["depth"],
                    "function": function,
                    "timestamp": entry["timestamp"],
                    "inputs": None,
                    "output": None,
                    "evaluation_runs": [],
                    "span_type": entry.get("span_type", "span")
                }
                active_functions.append(function)
                
            elif entry["type"] == "exit" and function in active_functions:
                # Complete function entry
                current_entry = function_entries[function]
                current_entry["duration"] = entry["timestamp"] - current_entry["timestamp"]
                condensed.append(current_entry)
                active_functions.remove(function)
                # del function_entries[function]
                
            # The OR condition is to handle the LLM client case.
            # LLM client is a special case where we exit the span, so when we attach evaluations to it, 
            # we have to check if the previous entry is an LLM call.
            elif function in active_functions or entry["type"] == "evaluation" and entries[i-1]["function"] == entry["function"]:
                # Update existing function entry with additional data
                current_entry = function_entries[function]
                
                if entry["type"] == "input" and entry["inputs"]:
                    current_entry["inputs"] = entry["inputs"]
                    
                if entry["type"] == "output" and entry["output"]:
                    current_entry["output"] = entry["output"]
                    
                if entry["type"] == "evaluation" and entry["evaluation_runs"]:
                    current_entry["evaluation_runs"] = entry["evaluation_runs"]

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

        # Calculate total token counts from LLM API calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        for entry in condensed_entries:
            if entry.get("span_type") == "llm" and isinstance(entry.get("output"), dict):
                usage = entry["output"].get("usage", {})
                # Handle OpenAI/Together format
                if "prompt_tokens" in usage:
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                # Handle Anthropic format
                elif "input_tokens" in usage:
                    total_prompt_tokens += usage.get("input_tokens", 0)
                    total_completion_tokens += usage.get("output_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)

        # Create trace document
        trace_data = {
            "trace_id": self.trace_id,
            "name": self.name,
            "project_name": self.project_name,
            "created_at": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration": total_duration,
            "token_counts": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            },
            "entries": condensed_entries,
            "empty_save": empty_save,
            "overwrite": overwrite
        }
        # Execute asynchrous evaluation in the background
        if not empty_save:  # Only send to RabbitMQ if the trace is not empty
            # Send trace data to evaluation queue via API
            try:
                response = requests.post(
                    JUDGMENT_TRACES_ADD_TO_EVAL_QUEUE_API_URL,
                    json=trace_data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.tracer.api_key}",
                        "X-Organization-Id": self.tracer.organization_id
                    },
                    verify=True
                )
                
                if response.status_code != HTTPStatus.OK:
                    warnings.warn(f"Failed to add trace to evaluation queue: {response.text}")
            except Exception as e:
                warnings.warn(f"Error sending trace to evaluation queue: {str(e)}")
        
        self.trace_manager_client.save_trace(trace_data, empty_save)

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
        organization_id: str = os.getenv("JUDGMENT_ORG_ID")):
        if not hasattr(self, 'initialized'):
            if not api_key:
                raise ValueError("Tracer must be configured with a Judgment API key")
            
            if not organization_id:
                raise ValueError("Tracer must be configured with an Organization ID")
            
            self.api_key: str = api_key
            self.project_name: str = project_name
            self.client: JudgmentClient = JudgmentClient(judgment_api_key=api_key)
            self.organization_id: str = organization_id
            self.depth: int = 0
            self._current_trace: Optional[str] = None
            self.rules: List[Rule] = rules or []  # Store rules at tracer level
            self.initialized: bool = True
        elif hasattr(self, 'project_name') and self.project_name != project_name:
            warnings.warn(
                f"Attempting to initialize Tracer with project_name='{project_name}' but it was already initialized with "
                f"project_name='{self.project_name}'. Due to the singleton pattern, the original project_name will be used. "
                "To use a different project name, ensure the first Tracer initialization uses the desired project name.",
                RuntimeWarning
            )
        
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

        # Reset depth if this is a top-level trace (not nested in another trace)
        is_top_level = self._current_trace is None
        if is_top_level:
            self.depth = 0  # Reset depth counter for new top-level traces

        trace = TraceClient(
            self, 
            trace_id, 
            name, 
            project_name=project, 
            overwrite=overwrite,
            rules=self.rules  # Pass combined rules to the trace client
        )
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
                # Reset depth to 0 if this was a top-level trace
                if is_top_level:
                    self.depth = 0
                
    def get_current_trace(self) -> Optional[TraceClient]:
        """
        Get the current trace context
        """
        return self._current_trace    

    def observe(self, func=None, *, name=None, span_type: SpanType = "span", project_name: str = None, overwrite: bool = False):
        """
        Decorator to trace function execution with detailed entry/exit information.
        
        Args:
            func: The function to decorate
            name: Optional custom name for the span (defaults to function name)
            span_type: Type of span (default "span")
            project_name: Optional project name override
            overwrite: Whether to overwrite existing traces
        """
        if func is None:
            return lambda f: self.observe(f, name=name, span_type=span_type, project_name=project_name, overwrite=overwrite)
        
        # Use provided name or fall back to function name
        span_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # If there's already a trace, use it. Otherwise create a new one
                if self._current_trace:
                    trace = self._current_trace
                else:
                    trace_id = str(uuid.uuid4())
                    trace_name = func.__name__
                    project = project_name if project_name is not None else self.project_name
                    trace = TraceClient(self, trace_id, trace_name, project_name=project, overwrite=overwrite, rules=self.rules)
                    self._current_trace = trace
                    # Only save empty trace for the root call
                    trace.save(empty_save=True, overwrite=overwrite)

                try:
                    with trace.span(span_name, span_type=span_type) as span:
                        # Record inputs
                        span.record_input({
                            'args': str(args),
                            'kwargs': kwargs
                        })
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Record output
                        span.record_output(result)
                        
                        return result
                finally:
                    # Only save and cleanup if this is the root observe call
                    if self.depth == 0:
                        trace.save(empty_save=False, overwrite=overwrite)
                        self._current_trace = None
                    
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # If there's already a trace, use it. Otherwise create a new one
                if self._current_trace:
                    trace = self._current_trace
                else:
                    trace_id = str(uuid.uuid4())
                    trace_name = func.__name__
                    project = project_name if project_name is not None else self.project_name
                    trace = TraceClient(self, trace_id, trace_name, project_name=project, overwrite=overwrite, rules=self.rules)
                    self._current_trace = trace
                    # Only save empty trace for the root call
                    trace.save(empty_save=True, overwrite=overwrite)
                
                try:
                    with trace.span(span_name, span_type=span_type) as span:
                        # Record inputs
                        span.record_input({
                            'args': str(args),
                            'kwargs': kwargs
                        })
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Record output
                        span.record_output(result)
                        
                        return result
                finally:
                    # Only save and cleanup if this is the root observe call
                    if self.depth == 0:
                        trace.save(empty_save=False, overwrite=overwrite)
                        self._current_trace = None
                    
            return wrapper
        
    def score(self, func=None, scorers: List[Union[APIJudgmentScorer, JudgevalScorer]] = None, model: str = None, log_results: bool = True, *, name: str = None, span_type: SpanType = "span"):
        """
        Decorator to trace function execution with detailed entry/exit information.
        """
        if func is None:
            return lambda f: self.observe(f, name=name, span_type=span_type)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if self._current_trace:
                    self._current_trace.async_evaluate(scorers=[scorers], input=args, actual_output=kwargs, model=model, log_results=log_results)
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self._current_trace:
                    self._current_trace.async_evaluate(scorers=[scorers], input=args, actual_output=kwargs, model="gpt-4o-mini", log_results=True)
            return wrapper
        


def wrap(client: Any) -> Any:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, and Anthropic clients.
    """
    # Get the appropriate configuration for this client type
    span_name, original_create = _get_client_config(client)
    
    def traced_create(*args, **kwargs):
        # Get the current tracer instance (might be created after client was wrapped)
        tracer = Tracer._instance
        
        # Skip tracing if no tracer exists or no active trace
        if not tracer or not tracer._current_trace:
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

class JudgevalCallbackHandler(BaseCallbackHandler):
    def __init__(self, trace_client: TraceClient):
        self.trace_client = trace_client
        self.openai_count = 1

    def start_span(self, name: str, span_type: SpanType = "span"):
        start_time = time.time()
        
        # Record span entry
        self.trace_client.add_entry(TraceEntry(
            type="enter",
            function=name,
            depth=self.trace_client.tracer.depth,
            message=name,
            timestamp=start_time,
            span_type=span_type
        ))
        
        self.trace_client.tracer.depth += 1
        self.trace_client.prev_span = self.trace_client._current_span
        self.trace_client._current_span = name
        self._start_time = start_time
    
    def end_span(self, name: str, span_type: SpanType = "span"):
        self.trace_client.tracer.depth -= 1
        duration = time.time() - self._start_time
        
        # Record span exit
        self.trace_client.add_entry(TraceEntry(
            type="exit",
            function=name,
            depth=self.trace_client.tracer.depth,
            message=f"← {name}",
            timestamp=time.time(),
            duration=duration,
            span_type=span_type
        ))
        self.trace_client._current_span = self.trace_client.prev_span

    def on_retriever_start(
        self,
        serialized: Optional[dict[str, Any]],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        name = "RETRIEVER_CALL"
        if serialized and "name" in serialized:
            name = f"RETRIEVER_{serialized['name'].upper()}"
        
        self.start_span(name, span_type="retriever")
        self.trace_client.record_input({
            'query': query,
            'tags': tags,
            'metadata': metadata,
            'kwargs': kwargs
        })

    def on_retriever_end(
        self, 
        documents: Sequence[Document], 
        *, 
        run_id: UUID, 
        parent_run_id: Optional[UUID] = None, 
        **kwargs: Any
    ) -> Any:
        # Process the retrieved documents into a format suitable for logging
        doc_summary = []
        for i, doc in enumerate(documents):
            # Extract key information from each document
            doc_data = {
                "index": i,
                "page_content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "metadata": doc.metadata
            }
            doc_summary.append(doc_data)
        
        # Record the document data
        self.trace_client.record_output({
            "document_count": len(documents),
            "documents": doc_summary
        })
        
        # End the retriever span
        self.end_span(self.trace_client._current_span, span_type="retriever")

    def on_tool_start(
        self,
        serialized: Optional[dict[str, Any]],
        input_str: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        name = serialized["name"]
        self.start_span(name, span_type="tool")
        self.trace_client.record_input({
            'args': input_str,
            'kwargs': kwargs
        })

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        self.trace_client.record_output(output)
        self.end_span(self.trace_client._current_span, span_type="tool")

    def on_agent_action (self, action: AgentAction, **kwargs: Any) -> Any:
        print(f"Agent action: {action}")

    def on_agent_finish(
            self,
            finish: AgentFinish,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[list[str]] = None,
            **kwargs: Any,
        ) -> None:
            print(f"Agent action: {finish}")

    def on_llm_start(
        self,
        serialized: Optional[dict[str, Any]],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:    
        name = "LLM call"
        self.start_span(name, span_type="llm")
        self.trace_client.record_input({
            'args': prompts,
            'kwargs': kwargs
        })

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any):
        self.trace_client.record_output(response.generations[0][0].text)
        self.end_span(self.trace_client._current_span, span_type="llm")

    def on_chat_model_start(
        self,
        serialized: Optional[dict[str, Any]],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:

        if "openai" in serialized["id"]:
            name = f"OPENAI_API_CALL_{self.openai_count}"
            self.openai_count += 1
        elif "anthropic" in serialized["id"]:
            name = "ANTHROPIC_API_CALL"
        elif "together" in serialized["id"]:
            name = "TOGETHER_API_CALL"
        else:
            name = "LLM call"

        self.start_span(name, span_type="llm")
        self.trace_client.record_input({
            'args': str(messages),
            'kwargs': kwargs
        })
