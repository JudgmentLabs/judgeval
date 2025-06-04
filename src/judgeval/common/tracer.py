"""
Tracing system for judgeval that allows for function tracing using decorators.
"""
# Standard library imports
import asyncio
import functools
import inspect
import os
import site
import sysconfig
import threading
import time
import traceback
import uuid
import warnings
import contextvars
import sys
import json
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
    Union,
    AsyncGenerator,
    TypeAlias,
)
from rich import print as rprint
import types

# Third-party imports
import requests
from litellm import cost_per_token as _original_cost_per_token
from rich import print as rprint
from openai import OpenAI, AsyncOpenAI
from together import Together, AsyncTogether
from anthropic import Anthropic, AsyncAnthropic
from google import genai

# Local application/library-specific imports
from judgeval.constants import (
    JUDGMENT_TRACES_ADD_ANNOTATION_API_URL,
    JUDGMENT_TRACES_SAVE_API_URL,
    JUDGMENT_TRACES_FETCH_API_URL,
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_QUEUE,
    JUDGMENT_TRACES_DELETE_API_URL,
    JUDGMENT_PROJECT_DELETE_API_URL,
)
from judgeval.data import Example, Trace, TraceSpan, TraceUsage
from judgeval.scorers import APIJudgmentScorer, JudgevalScorer
from judgeval.rules import Rule
from judgeval.evaluation_run import EvaluationRun
from judgeval.common.utils import ExcInfo, validate_api_key
from judgeval.common.exceptions import JudgmentAPIError

# Standard library imports needed for the new class
import concurrent.futures
from collections.abc import Iterator, AsyncIterator # Add Iterator and AsyncIterator

# Define context variables for tracking the current trace and the current span within a trace
current_trace_var = contextvars.ContextVar[Optional['TraceClient']]('current_trace', default=None)
current_span_var = contextvars.ContextVar('current_span', default=None) # ContextVar for the active span name

# Define type aliases for better code readability and maintainability
ApiClient: TypeAlias = Union[OpenAI, Together, Anthropic, AsyncOpenAI, AsyncAnthropic, AsyncTogether, genai.Client, genai.client.AsyncClient]  # Supported API clients
SpanType = Literal['span', 'tool', 'llm', 'evaluation', 'chain']

# --- Evaluation Config Dataclass (Moved from langgraph.py) ---
@dataclass
class EvaluationConfig:
    """Configuration for triggering an evaluation from the handler."""
    scorers: List[Union[APIJudgmentScorer, JudgevalScorer]]
    example: Example
    model: Optional[str] = None
    log_results: Optional[bool] = True
# --- End Evaluation Config Dataclass ---

# Temporary as a POC to have log use the existing annotations feature until log endpoints are ready
@dataclass
class TraceAnnotation:
    """Represents a single annotation for a trace span."""
    span_id: str
    text: str
    label: str
    score: int

    def to_dict(self) -> dict:
        """Convert the annotation to a dictionary format for storage/transmission."""
        return {
            "span_id": self.span_id,
            "annotation": {
                "text": self.text,
                "label": self.label,
                "score": self.score
            }
        }
    
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

    def save_trace(self, trace_data: dict, offline_mode: bool = False):
        """
        Saves a trace to the Judgment Supabase and optionally to S3 if configured.

        Args:
            trace_data: The trace data to save
            NOTE we save empty traces in order to properly handle async operations; we need something in the DB to associate the async results with
        """
        # Save to Judgment API
        
        def fallback_encoder(obj):
            """
            Custom JSON encoder fallback.
            Tries to use obj.__repr__(), then str(obj) if that fails or for a simpler string.
            You can choose which one you prefer or try them in sequence.
            """
            try:
                # Option 1: Prefer __repr__ for a more detailed representation
                return repr(obj)
            except Exception:
                # Option 2: Fallback to str() if __repr__ fails or if you prefer str()
                try:
                    return str(obj)
                except Exception as e:
                    # If both fail, you might return a placeholder or re-raise
                    return f"<Unserializable object of type {type(obj).__name__}: {e}>"
        
        serialized_trace_data = json.dumps(trace_data, default=fallback_encoder)

        response = requests.post(
            JUDGMENT_TRACES_SAVE_API_URL,
            data=serialized_trace_data,
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
        
        if not offline_mode and "ui_results_url" in response.json():
            pretty_str = f"\n🔍 You can view your trace data here: [rgb(106,0,255)][link={response.json()['ui_results_url']}]View Trace[/link]\n"
            rprint(pretty_str)

    ## TODO: Should have a log endpoint, endpoint should also support batched payloads
    def save_annotation(self, annotation: TraceAnnotation):
        json_data = {
            "span_id": annotation.span_id,
            "annotation": {
                "text": annotation.text,
                "label": annotation.label,
                "score": annotation.score
            }
        }       

        response = requests.post(
            JUDGMENT_TRACES_ADD_ANNOTATION_API_URL,
            json=json_data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.judgment_api_key}',
                'X-Organization-Id': self.organization_id
            },
            verify=True
        )
        
        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to save annotation: {response.text}")
        
        return response.json()

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
        project_name: str = None,
        overwrite: bool = False,
        rules: Optional[List[Rule]] = None,
        enable_monitoring: bool = True,
        enable_evaluations: bool = True,
        parent_trace_id: Optional[str] = None,
        parent_name: Optional[str] = None
    ):
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.project_name = project_name or str(uuid.uuid4())
        self.overwrite = overwrite
        self.tracer = tracer
        self.rules = rules or []
        self.enable_monitoring = enable_monitoring
        self.enable_evaluations = enable_evaluations
        self.parent_trace_id = parent_trace_id
        self.parent_name = parent_name
        self.trace_spans: List[TraceSpan] = []
        self.span_id_to_span: Dict[str, TraceSpan] = {}
        self.evaluation_runs: List[EvaluationRun] = []
        self.annotations: List[TraceAnnotation] = []
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
            
        span = TraceSpan(
            span_id=span_id,
            trace_id=self.trace_id,
            depth=current_depth,
            message=name,
            created_at=start_time,
            span_type=span_type,
            parent_span_id=parent_span_id,
            function=name,
        )
        self.add_span(span)
        
        try:
            yield self
        finally:
            duration = time.time() - start_time
            span.duration = duration
            # Clean up depth tracking for this span_id
            if span_id in self._span_depths:
                del self._span_depths[span_id]
            # Reset context var
            current_span_var.reset(token)

    def async_evaluate(
        self,
        scorers: List[Union[APIJudgmentScorer, JudgevalScorer]],
        example: Optional[Example] = None,
        input: Optional[str] = None,
        actual_output: Optional[Union[str, List[str]]] = None,
        expected_output: Optional[Union[str, List[str]]] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        tools_called: Optional[List[str]] = None,
        expected_tools: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        span_id: Optional[str] = None, # <<< ADDED optional span_id parameter
        log_results: Optional[bool] = True
    ):
        if not self.enable_evaluations:
            return
        
        start_time = time.time()  # Record start time

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
        
        # If example is not provided, create one from the individual parameters
        if example is None:
            # Check if any of the individual parameters are provided
            if any(param is not None for param in [input, actual_output, expected_output, context, 
                                                retrieval_context, tools_called, expected_tools, 
                                                additional_metadata]):
                example = Example(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
                    tools_called=tools_called,
                    expected_tools=expected_tools,
                    additional_metadata=additional_metadata,
                )
            else:
                raise ValueError("Either 'example' or at least one of the individual parameters (input, actual_output, etc.) must be provided")
        
        # Check examples before creating evaluation run
        
        # check_examples([example], scorers)
        
        # --- Modification: Capture span_id immediately ---
        # span_id_at_eval_call = current_span_var.get()
        # print(f"[TraceClient.async_evaluate] Captured span ID at eval call: {span_id_at_eval_call}")
        # Prioritize explicitly passed span_id, fallback to context var
        current_span_ctx_var = current_span_var.get()
        span_id_to_use = span_id if span_id is not None else current_span_ctx_var if current_span_ctx_var is not None else self.tracer.get_current_span()
        # print(f"[TraceClient.async_evaluate] Using span_id: {span_id_to_use}")
        # --- End Modification ---

        # Combine the trace-level rules with any evaluation-specific rules)
        eval_run = EvaluationRun(
            organization_id=self.tracer.organization_id,
            log_results=log_results,
            project_name=self.project_name,
            eval_name=f"{self.name.capitalize()}-"
                f"{span_id_to_use}-" # Keep original eval name format using context var if available
                f"[{','.join(scorer.score_type.capitalize() for scorer in scorers)}]",
            examples=[example],
            scorers=scorers,
            model=model,
            metadata={},
            judgment_api_key=self.tracer.api_key,
            override=self.overwrite,
            trace_span_id=span_id_to_use, # Pass the determined ID
            rules=self.rules # Use the combined rules
        )
        
        self.add_eval_run(eval_run, start_time)  # Pass start_time to record_evaluation
            
    def add_eval_run(self, eval_run: EvaluationRun, start_time: float):
        # --- Modification: Use span_id from eval_run --- 
        current_span_id = eval_run.trace_span_id # Get ID from the eval_run object
        # print(f"[TraceClient.add_eval_run] Using span_id from eval_run: {current_span_id}")
        # --- End Modification ---

        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.evaluation_runs.append(eval_run)
            span.has_evaluation = True  # Set the has_evaluation flag
        self.evaluation_runs.append(eval_run)

    def add_annotation(self, annotation: TraceAnnotation):
       """Add an annotation to this trace context"""
       self.annotations.append(annotation)
       return self
    
    def record_input(self, inputs: dict):
        current_span_id = current_span_var.get()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            # Ignore self parameter
            if "self" in inputs:
                del inputs["self"]
            span.inputs = inputs
    
    def record_agent_name(self, agent_name: str):
        current_span_id = current_span_var.get()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.agent_name = agent_name

    def record_state_before(self, state: dict):
        """Records the agent's state before a tool execution on the current span.

        Args:
            state: A dictionary representing the agent's state.
        """
        current_span_id = current_span_var.get()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.state_before = state
            
    def record_state_after(self, state: dict):
        """Records the agent's state after a tool execution on the current span.

        Args:
            state: A dictionary representing the agent's state.
        """
        current_span_id = current_span_var.get()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.state_after = state

    async def _update_coroutine(self, span: TraceSpan, coroutine: Any, field: str):
        """Helper method to update the output of a trace entry once the coroutine completes"""
        try:
            result = await coroutine
            setattr(span, field, result)
            return result
        except Exception as e:
            setattr(span, field, f"Error: {str(e)}")
            raise

    def record_output(self, output: Any):
        current_span_id = current_span_var.get()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.output = "<pending>" if inspect.iscoroutine(output) else output
            
            if inspect.iscoroutine(output):
                asyncio.create_task(self._update_coroutine(span, output, "output"))

            return span # Return the created entry
        # Removed else block - original didn't have one
        return None # Return None if no span_id found
    
    def record_usage(self, usage: TraceUsage):
        current_span_id = current_span_var.get()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.usage = usage
            
            return span # Return the created entry
        # Removed else block - original didn't have one
        return None # Return None if no span_id found
    
    def record_error(self, error: Dict[str, Any]):
        current_span_id = current_span_var.get()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.error = error
            return span
        return None
    
    def add_span(self, span: TraceSpan):
        """Add a trace span to this trace context"""
        self.trace_spans.append(span)
        self.span_id_to_span[span.span_id] = span
        return self
        
    def print(self):
        """Print the complete trace with proper visual structure"""
        for span in self.trace_spans:
            span.print_span()
            
    def get_duration(self) -> float:
        """
        Get the total duration of this trace
        """
        return time.time() - self.start_time

    def save(self, overwrite: bool = False) -> Tuple[str, dict]:
        """
        Save the current trace to the database.
        Returns a tuple of (trace_id, trace_data) where trace_data is the trace data that was saved.
        """
        # Calculate total elapsed time
        total_duration = self.get_duration()
        # Create trace document - Always use standard keys for top-level counts
        trace_data = {
            "trace_id": self.trace_id,
            "name": self.name,
            "project_name": self.project_name,
            "created_at": datetime.utcfromtimestamp(self.start_time).isoformat(),
            "duration": total_duration,
            "trace_spans": [span.model_dump() for span in self.trace_spans],
            "evaluation_runs": [run.model_dump() for run in self.evaluation_runs],
            "overwrite": overwrite,
            "offline_mode": self.tracer.offline_mode,
            "parent_trace_id": self.parent_trace_id,
            "parent_name": self.parent_name
        }        
        # --- Log trace data before saving ---
        self.trace_manager_client.save_trace(trace_data, offline_mode=self.tracer.offline_mode)

        # upload annotations
        # TODO: batch to the log endpoint
        for annotation in self.annotations:
            self.trace_manager_client.save_annotation(annotation)

        return self.trace_id, trace_data

    def delete(self):
        return self.trace_manager_client.delete_trace(self.trace_id)
    
def _capture_exception_for_trace(current_trace: Optional['TraceClient'], exc_info: ExcInfo):
    if not current_trace:
        return

    exc_type, exc_value, exc_traceback_obj = exc_info
    formatted_exception = {
        "type": exc_type.__name__ if exc_type else "UnknownExceptionType",
        "message": str(exc_value) if exc_value else "No exception message",
        "traceback": traceback.format_tb(exc_traceback_obj) if exc_traceback_obj else []
    }
    
    # This is where we specially handle exceptions that we might want to collect additional data for.
    # When we do this, always try checking the module from sys.modules instead of importing. This will
    # Let us support a wider range of exceptions without needing to import them for all clients.
    
    # Most clients (requests, httpx, urllib) support the standard format of exposing error.request.url and error.response.status_code 
    # The alternative is to hand select libraries we want from sys.modules and check for them:
    # As an example:  requests_module = sys.modules.get("requests", None) // then do things with requests_module;

     # General HTTP Like errors
    try:
        url = getattr(getattr(exc_value, "request", None), "url", None)
        status_code = getattr(getattr(exc_value, "response", None), "status_code", None)
        if status_code:
            formatted_exception["http"] = {
                "url": url if url else "Unknown URL",
                "status_code": status_code if status_code else None,
            }
    except Exception as e:
        pass

    current_trace.record_error(formatted_exception)
class _DeepTracer:
    _instance: Optional["_DeepTracer"] = None
    _lock: threading.Lock = threading.Lock()
    _refcount: int = 0
    _span_stack: contextvars.ContextVar[List[Dict[str, Any]]] = contextvars.ContextVar("_deep_profiler_span_stack", default=[])
    _skip_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar("_deep_profiler_skip_stack", default=[])
    _original_sys_trace: Optional[Callable] = None
    _original_threading_trace: Optional[Callable] = None

    def _get_qual_name(self, frame) -> str:
        func_name = frame.f_code.co_name
        module_name = frame.f_globals.get("__name__", "unknown_module")
        
        try:
            func = frame.f_globals.get(func_name)
            if func is None:
                return f"{module_name}.{func_name}"
            if hasattr(func, "__qualname__"):
                 return f"{module_name}.{func.__qualname__}"
        except Exception:
            return f"{module_name}.{func_name}"
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def _should_trace(self, frame):
        # Skip stack is maintained by the tracer as an optimization to skip earlier
        # frames in the call stack that we've already determined should be skipped
        skip_stack = self._skip_stack.get()
        if len(skip_stack) > 0:
            return False
        
        func_name = frame.f_code.co_name
        module_name = frame.f_globals.get("__name__", None)

        func = frame.f_globals.get(func_name)
        if func and (hasattr(func, '_judgment_span_name') or hasattr(func, '_judgment_span_type')):
            return False

        if (
            not module_name
            or func_name.startswith("<") # ex: <listcomp>
            or func_name.startswith("__") and func_name != "__call__" # dunders
            or not self._is_user_code(frame.f_code.co_filename)
        ):
            return False
                    
        return True
    
    @functools.cache
    def _is_user_code(self, filename: str):
        return bool(filename) and not filename.startswith("<") and not os.path.realpath(filename).startswith(_TRACE_FILEPATH_BLOCKLIST)

    def _cooperative_sys_trace(self, frame: types.FrameType, event: str, arg: Any):
        """Cooperative trace function for sys.settrace that chains with existing tracers."""
        # First, call the original sys trace function if it exists
        original_result = None
        if self._original_sys_trace:
            try:
                original_result = self._original_sys_trace(frame, event, arg)
            except Exception:
                # If the original tracer fails, continue with our tracing
                pass
        
        # Then do our own tracing
        our_result = self._trace(frame, event, arg, self._cooperative_sys_trace)
        
        # Return our tracer to continue tracing, but respect the original's decision
        # If the original tracer returned None (stop tracing), we should respect that
        if original_result is None and self._original_sys_trace:
            return None
        
        return our_result or original_result
    
    def _cooperative_threading_trace(self, frame: types.FrameType, event: str, arg: Any):
        """Cooperative trace function for threading.settrace that chains with existing tracers."""
        # First, call the original threading trace function if it exists
        original_result = None
        if self._original_threading_trace:
            try:
                original_result = self._original_threading_trace(frame, event, arg)
            except Exception:
                # If the original tracer fails, continue with our tracing
                pass
        
        # Then do our own tracing
        our_result = self._trace(frame, event, arg, self._cooperative_threading_trace)
        
        # Return our tracer to continue tracing, but respect the original's decision
        # If the original tracer returned None (stop tracing), we should respect that
        if original_result is None and self._original_threading_trace:
            return None
        
        return our_result or original_result
    
    def _trace(self, frame: types.FrameType, event: str, arg: Any, continuation_func: Callable):
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False

        if not self._should_trace(frame):
            return
        
        if event not in ("call", "return", "exception"):
            return
        
        current_trace = current_trace_var.get()
        if not current_trace:
            return
        
        parent_span_id = current_span_var.get()
        if not parent_span_id:
            return

        qual_name = self._get_qual_name(frame)
        instance_name = None
        if 'self' in frame.f_locals:
            instance = frame.f_locals['self']
            class_name = instance.__class__.__name__
            class_identifiers = getattr(Tracer._instance, 'class_identifiers', {})
            instance_name = get_instance_prefixed_name(instance, class_name, class_identifiers)
        skip_stack = self._skip_stack.get()
        
        if event == "call":
            # If we have entries in the skip stack and the current qual_name matches the top entry,
            # push it again to track nesting depth and skip
            # As an optimization, we only care about duplicate qual_names.
            if skip_stack:
                if qual_name == skip_stack[-1]:
                    skip_stack.append(qual_name)
                    self._skip_stack.set(skip_stack)
                return
            
            should_trace = self._should_trace(frame)
            
            if not should_trace:
                if not skip_stack:
                    self._skip_stack.set([qual_name])
                return
        elif event == "return":
            # If we have entries in skip stack and current qual_name matches the top entry,
            # pop it to track exiting from the skipped section
            if skip_stack and qual_name == skip_stack[-1]:
                skip_stack.pop()
                self._skip_stack.set(skip_stack)
                return
            
            if skip_stack:
                return
            
        span_stack = self._span_stack.get()
        if event == "call":
            if not self._should_trace(frame):
                return
                
            span_id = str(uuid.uuid4())
            
            parent_depth = current_trace._span_depths.get(parent_span_id, 0)
            depth = parent_depth + 1
            
            current_trace._span_depths[span_id] = depth
            
            start_time = time.time()
            
            span_stack.append({
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "function": qual_name,
                "start_time": start_time
            })
            self._span_stack.set(span_stack)
            
            token = current_span_var.set(span_id)
            frame.f_locals["_judgment_span_token"] = token
            
            span = TraceSpan(
                span_id=span_id,
                trace_id=current_trace.trace_id,
                depth=depth,
                message=qual_name,
                created_at=start_time,
                span_type="span",
                parent_span_id=parent_span_id,
                function=qual_name,
                agent_name=instance_name
            )
            current_trace.add_span(span)
            
            inputs = {}
            try:
                args_info = inspect.getargvalues(frame)
                for arg in args_info.args:
                    try:
                        inputs[arg] = args_info.locals.get(arg)
                    except:
                        inputs[arg] = "<<Unserializable>>"
                current_trace.record_input(inputs)
            except Exception as e:
                current_trace.record_input({
                    "error": str(e)
                })
                
        elif event == "return":
            if not span_stack:
                return
                
            current_id = current_span_var.get()
            
            span_data = None
            for i, entry in enumerate(reversed(span_stack)):
                if entry["span_id"] == current_id:
                    span_data = span_stack.pop(-(i+1))
                    self._span_stack.set(span_stack)
                    break
            
            if not span_data:
                return
                
            start_time = span_data["start_time"]
            duration = time.time() - start_time
            
            current_trace.span_id_to_span[span_data["span_id"]].duration = duration

            if arg is not None:
                # exception handling will take priority. 
                current_trace.record_output(arg)
            
            if span_data["span_id"] in current_trace._span_depths:
                del current_trace._span_depths[span_data["span_id"]]
                
            if span_stack:
                current_span_var.set(span_stack[-1]["span_id"])
            else:
                current_span_var.set(span_data["parent_span_id"])
            
            if "_judgment_span_token" in frame.f_locals:
                current_span_var.reset(frame.f_locals["_judgment_span_token"])

        elif event == "exception":
            exc_type = arg[0]
            if issubclass(exc_type, (StopIteration, StopAsyncIteration, GeneratorExit)):
                return
            _capture_exception_for_trace(current_trace, arg)
            
        
        return continuation_func
    
    def __enter__(self):
        with self._lock:
            self._refcount += 1
            if self._refcount == 1:
                # Store the existing trace functions before setting ours
                self._original_sys_trace = sys.gettrace()
                self._original_threading_trace = threading.gettrace()
                
                self._skip_stack.set([])
                self._span_stack.set([])
                
                sys.settrace(self._cooperative_sys_trace)
                threading.settrace(self._cooperative_threading_trace)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._refcount -= 1
            if self._refcount == 0:
                # Restore the original trace functions instead of setting to None
                sys.settrace(self._original_sys_trace)
                threading.settrace(self._original_threading_trace)
                
                # Clean up the references
                self._original_sys_trace = None
                self._original_threading_trace = None


def log(self, message: str, level: str = "info"):
        """ Log a message with the span context """
        current_trace = current_trace_var.get()
        if current_trace:
            current_trace.log(message, level)
        else:
            print(f"[{level}] {message}")
        current_trace.record_output({"log": message})
    
class Tracer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, 
        api_key: str = os.getenv("JUDGMENT_API_KEY"), 
        project_name: str = None,
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
        offline_mode: bool = False,
        deep_tracing: bool = True  # Deep tracing is enabled by default
        ):
        if not hasattr(self, 'initialized'):
            if not api_key:
                raise ValueError("Tracer must be configured with a Judgment API key")
            
            result, response = validate_api_key(api_key)
            if not result:
                raise JudgmentAPIError(f"Issue with passed in Judgment API key: {response}")
            
            if not organization_id:
                raise ValueError("Tracer must be configured with an Organization ID")
            if use_s3 and not s3_bucket_name:
                raise ValueError("S3 bucket name must be provided when use_s3 is True")
            
            self.api_key: str = api_key
            self.project_name: str = project_name or str(uuid.uuid4())
            self.organization_id: str = organization_id
            self._current_trace: Optional[str] = None
            self._active_trace_client: Optional[TraceClient] = None # Add active trace client attribute
            self.rules: List[Rule] = rules or []  # Store rules at tracer level
            self.traces: List[Trace] = []
            self.initialized: bool = True
            self.enable_monitoring: bool = enable_monitoring
            self.enable_evaluations: bool = enable_evaluations
            self.class_identifiers: Dict[str, str] = {}  # Dictionary to store class identifiers

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
            self.offline_mode: bool = offline_mode
            self.deep_tracing: bool = deep_tracing  # NEW: Store deep tracing setting

        elif hasattr(self, 'project_name') and self.project_name != project_name:
            warnings.warn(
                f"Attempting to initialize Tracer with project_name='{project_name}' but it was already initialized with "
                f"project_name='{self.project_name}'. Due to the singleton pattern, the original project_name will be used. "
                "To use a different project name, ensure the first Tracer initialization uses the desired project name.",
                RuntimeWarning
            )

    def set_current_span(self, span_id: str):
        self.current_span_id = span_id
    
    def get_current_span(self) -> Optional[str]:
        return getattr(self, 'current_span_id', None)
    
    def set_current_trace(self, trace: TraceClient):
        """
        Set the current trace context in contextvars
        """
        current_trace_var.set(trace)
    
    def get_current_trace(self) -> Optional[TraceClient]:
        """
        Get the current trace context.

        Tries to get the trace client from the context variable first.
        If not found (e.g., context lost across threads/tasks),
        it falls back to the active trace client managed by the callback handler.
        """
        trace_from_context = current_trace_var.get()
        if trace_from_context:
            return trace_from_context
        
        # Fallback: Check the active client potentially set by a callback handler
        if hasattr(self, '_active_trace_client') and self._active_trace_client:
            # warnings.warn("Falling back to _active_trace_client in get_current_trace. ContextVar might be lost.", RuntimeWarning)
            return self._active_trace_client
            
        # If neither is available
        # warnings.warn("No current trace found in context variable or active client fallback.", RuntimeWarning)
        return None
        
    def get_active_trace_client(self) -> Optional[TraceClient]:
        """Returns the TraceClient instance currently marked as active by the handler."""
        return self._active_trace_client


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


    def log(self, msg: str, label: str = "log", score: int = 1):
        """Log a message with the current span context"""
        current_span_id = current_span_var.get()
        current_trace = current_trace_var.get()
        if current_span_id:
            annotation = TraceAnnotation(
                span_id=current_span_id,
                text=msg,
                label=label,
                score=score
            )

            current_trace.add_annotation(annotation)

        rprint(f"[bold]{label}:[/bold] {msg}")
    
    def identify(self, identifier: str, track_state: bool = False, track_attributes: Optional[List[str]] = None, field_mappings: Optional[Dict[str, str]] = None):
        """
        Class decorator that associates a class with a custom identifier and enables state tracking.
        
        This decorator creates a mapping between the class name and the provided
        identifier, which can be useful for tagging, grouping, or referencing
        classes in a standardized way. It also enables automatic state capture
        for instances of the decorated class when used with tracing.
        
        Args:
            identifier: The identifier to associate with the decorated class.
                    This will be used as the instance name in traces.
            track_state: Whether to automatically capture the state (attributes) 
                        of instances before and after function execution. Defaults to False.
            track_attributes: Optional list of specific attribute names to track.
                            If None, all non-private attributes (not starting with '_') 
                            will be tracked when track_state=True.
            field_mappings: Optional dictionary mapping internal attribute names to 
                        display names in the captured state. For example:
                        {"system_prompt": "instructions"} will capture the 
                        'instructions' attribute as 'system_prompt' in the state.
            
        Example:
            @tracer.identify(identifier="user_model", track_state=True, track_attributes=["name", "age"], field_mappings={"system_prompt": "instructions"})
            class User:
                # Class implementation
        """
        def decorator(cls):
            class_name = cls.__name__
            self.class_identifiers[class_name] = {
                "identifier": identifier,
                "track_state": track_state,
                "track_attributes": track_attributes,
                "field_mappings": field_mappings or {}
            }
            return cls
        
        return decorator
    
    def _capture_instance_state(self, instance: Any, class_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Capture the state of an instance based on class configuration.
        Args:
            instance: The instance to capture the state of.
            class_config: Configuration dictionary for state capture, 
                          expected to contain 'track_attributes' and 'field_mappings'.
        """
        track_attributes = class_config.get('track_attributes')
        field_mappings = class_config.get('field_mappings')
        
        if track_attributes:
        
            state = {attr: getattr(instance, attr, None) for attr in track_attributes}
        else:

            state = {k: v for k, v in instance.__dict__.items() if not k.startswith('_')}

        if field_mappings:
            state['field_mappings'] = field_mappings

        return state
        
        
    def _get_instance_state_if_tracked(self, args):
        """
        Extract instance state if the instance should be tracked.
        
        Returns the captured state dict if tracking is enabled, None otherwise.
        """
        if args and hasattr(args[0], '__class__'):
            instance = args[0]
            class_name = instance.__class__.__name__
            if (class_name in self.class_identifiers and 
                isinstance(self.class_identifiers[class_name], dict) and
                self.class_identifiers[class_name].get('track_state', False)):
                return self._capture_instance_state(instance, self.class_identifiers[class_name])
            
    def _conditionally_capture_and_record_state(self, trace_client_instance: TraceClient, args: tuple, is_before: bool):
        """Captures instance state if tracked and records it via the trace_client."""
        state = self._get_instance_state_if_tracked(args)
        if state:
            if is_before:
                trace_client_instance.record_state_before(state)
            else:
                trace_client_instance.record_state_after(state)
        
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
        original_span_name = name or func.__name__
        
        # Store custom attributes on the function object
        func._judgment_span_name = original_span_name
        func._judgment_span_type = span_type
        
        # Use the provided deep_tracing value or fall back to the tracer's default
        use_deep_tracing = deep_tracing if deep_tracing is not None else self.deep_tracing
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                nonlocal original_span_name
                class_name = None
                instance_name = None
                span_name = original_span_name
                agent_name = None

                if args and hasattr(args[0], '__class__'):
                    class_name = args[0].__class__.__name__
                    agent_name = get_instance_prefixed_name(args[0], class_name, self.class_identifiers)

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
                            inputs = combine_args_kwargs(func, args, kwargs)
                            span.record_input(inputs)
                            if agent_name:
                                span.record_agent_name(agent_name)

                            # Capture state before execution
                            self._conditionally_capture_and_record_state(span, args, is_before=True)
                            
                            if use_deep_tracing:
                                with _DeepTracer():
                                    result = await func(*args, **kwargs)
                            else:
                                try:
                                    result = await func(*args, **kwargs)
                                except Exception as e:
                                    _capture_exception_for_trace(current_trace, sys.exc_info())
                                    raise e
                                    
                            # Capture state after execution  
                            self._conditionally_capture_and_record_state(span, args, is_before=False)
                                                        
                            # Record output
                            span.record_output(result)
                        return result
                    finally:
                        # Save the completed trace
                        trace_id, trace = current_trace.save(overwrite=overwrite)
                        self.traces.append(trace)

                        # Reset trace context (span context resets automatically)
                        current_trace_var.reset(trace_token)
                else:
                    with current_trace.span(span_name, span_type=span_type) as span:
                        inputs = combine_args_kwargs(func, args, kwargs)
                        span.record_input(inputs)
                        if agent_name:
                            span.record_agent_name(agent_name)

                        # Capture state before execution
                        self._conditionally_capture_and_record_state(span, args, is_before=True)
                            
                        if use_deep_tracing:
                            with _DeepTracer():
                                result = await func(*args, **kwargs)
                        else:
                            try:
                                result = await func(*args, **kwargs)
                            except Exception as e:
                                _capture_exception_for_trace(current_trace, sys.exc_info())
                                raise e
                                
                        # Capture state after execution  
                        self._conditionally_capture_and_record_state(span, args, is_before=False)
                            
                        span.record_output(result)
                    return result
        
            return async_wrapper
        else:
            # Non-async function implementation with deep tracing
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal original_span_name
                class_name = None
                instance_name = None
                span_name = original_span_name
                agent_name = None
                if args and hasattr(args[0], '__class__'):
                    class_name = args[0].__class__.__name__
                    agent_name = get_instance_prefixed_name(args[0], class_name, self.class_identifiers)               
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
                            inputs = combine_args_kwargs(func, args, kwargs)
                            span.record_input(inputs)
                            if agent_name:
                                span.record_agent_name(agent_name)
                            # Capture state before execution
                            self._conditionally_capture_and_record_state(span, args, is_before=True)
                                    
                            if use_deep_tracing:
                                with _DeepTracer():
                                    result = func(*args, **kwargs)
                            else:
                                try:
                                    result = func(*args, **kwargs)
                                except Exception as e:
                                    _capture_exception_for_trace(current_trace, sys.exc_info())
                                    raise e
                                    
                            # Capture state after execution
                            self._conditionally_capture_and_record_state(span, args, is_before=False)

                            
                            # Record output
                            span.record_output(result)
                        return result
                    finally:
                        # Save the completed trace
                        trace_id, trace = current_trace.save(overwrite=overwrite)
                        self.traces.append(trace)

                        # Reset trace context (span context resets automatically)
                        current_trace_var.reset(trace_token)
                else:
                    with current_trace.span(span_name, span_type=span_type) as span:
                        
                        inputs = combine_args_kwargs(func, args, kwargs)
                        span.record_input(inputs)
                        if agent_name:
                            span.record_agent_name(agent_name)

                        # Capture state before execution
                        self._conditionally_capture_and_record_state(span, args, is_before=True)

                        if use_deep_tracing:
                            with _DeepTracer():
                                result = func(*args, **kwargs)
                        else:
                            try:
                                result = func(*args, **kwargs)
                            except Exception as e:
                                _capture_exception_for_trace(current_trace, sys.exc_info())
                                raise e
                            
                        # Capture state after execution
                        self._conditionally_capture_and_record_state(span, args, is_before=False)
                            
                        span.record_output(result)
                    return result
    
            return wrapper
        
    def async_evaluate(self, *args, **kwargs):
        if not self.enable_evaluations:
            return

        # --- Get trace_id passed explicitly (if any) ---
        passed_trace_id = kwargs.pop('trace_id', None) # Get and remove trace_id from kwargs

        # --- Get current trace from context FIRST ---
        current_trace = current_trace_var.get()

        # --- Fallback Logic: Use active client only if context var is empty ---
        if not current_trace:
            current_trace = self._active_trace_client # Use the fallback
        # --- End Fallback Logic ---

        if current_trace:
            # Pass the explicitly provided trace_id if it exists, otherwise let async_evaluate handle it
            # (Note: TraceClient.async_evaluate doesn't currently use an explicit trace_id, but this is for future proofing/consistency)
            if passed_trace_id:
                kwargs['trace_id'] = passed_trace_id # Re-add if needed by TraceClient.async_evaluate
            current_trace.async_evaluate(*args, **kwargs)
        else:
            warnings.warn("No trace found (context var or fallback), skipping evaluation") # Modified warning

def wrap(client: Any) -> Any:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, and Google GenAI clients.
    Patches both '.create' and Anthropic's '.stream' methods using a wrapper class.
    """
    span_name, original_create, original_responses_create, original_stream = _get_client_config(client)
    
    def _record_input_and_check_streaming(span, kwargs, is_responses=False):
        """Record input and check for streaming"""
        is_streaming = kwargs.get("stream", False)

            # Record input based on whether this is a responses endpoint
        if is_responses:
            span.record_input(kwargs)
        else:
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
            
        return is_streaming
    
    def _format_and_record_output(span, response, is_streaming, is_async, is_responses):
        """Format and record the output in the span"""
        if is_streaming:
            output_entry = span.record_output("<pending stream>")
            wrapper_func = _async_stream_wrapper if is_async else _sync_stream_wrapper
            return wrapper_func(response, client, output_entry)
        else:
            format_func = _format_response_output_data if is_responses else _format_output_data
            output, usage = format_func(client, response)
            span.record_output(output)
            span.record_usage(usage)
            return response
    
    # --- Traced Async Functions ---
    async def traced_create_async(*args, **kwargs):
        current_trace = current_trace_var.get()
        if not current_trace:
            return await original_create(*args, **kwargs)
        
        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(span, kwargs)
            
            try:
                response_or_iterator = await original_create(*args, **kwargs)
                return _format_and_record_output(span, response_or_iterator, is_streaming, True, False)
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e
    
    # Async responses for OpenAI clients
    async def traced_response_create_async(*args, **kwargs):
        current_trace = current_trace_var.get()
        if not current_trace:
            return await original_responses_create(*args, **kwargs)
        
        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(span, kwargs, is_responses=True)
            
            try:
                response_or_iterator = await original_responses_create(*args, **kwargs)
                return _format_and_record_output(span, response_or_iterator, is_streaming, True, True)
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e
    
    # Function replacing .stream() for async clients
    def traced_stream_async(*args, **kwargs):
        current_trace = current_trace_var.get()
        if not current_trace or not original_stream:
            return original_stream(*args, **kwargs)
        
        original_manager = original_stream(*args, **kwargs)
        return _TracedAsyncStreamManagerWrapper(
            original_manager=original_manager,
            client=client,
            span_name=span_name,
            trace_client=current_trace,
            stream_wrapper_func=_async_stream_wrapper,
            input_kwargs=kwargs
        )
    
    # --- Traced Sync Functions ---
    def traced_create_sync(*args, **kwargs):
        current_trace = current_trace_var.get()
        if not current_trace:
            return original_create(*args, **kwargs)
        
        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(span, kwargs)
            
            try:
                response_or_iterator = original_create(*args, **kwargs)
                return _format_and_record_output(span, response_or_iterator, is_streaming, False, False)
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e
    
    def traced_response_create_sync(*args, **kwargs):
        current_trace = current_trace_var.get()
        if not current_trace:
            return original_responses_create(*args, **kwargs)
        
        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(span, kwargs, is_responses=True)
            
            try:
                response_or_iterator = original_responses_create(*args, **kwargs)
                return _format_and_record_output(span, response_or_iterator, is_streaming, False, True)
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e
    
    # Function replacing sync .stream()
    def traced_stream_sync(*args, **kwargs):
        current_trace = current_trace_var.get()
        if not current_trace or not original_stream:
            return original_stream(*args, **kwargs)
        
        original_manager = original_stream(*args, **kwargs)
        return _TracedSyncStreamManagerWrapper(
            original_manager=original_manager,
            client=client,
            span_name=span_name,
            trace_client=current_trace,
            stream_wrapper_func=_sync_stream_wrapper,
            input_kwargs=kwargs
        )
    
    # --- Assign Traced Methods to Client Instance ---
    if isinstance(client, (AsyncOpenAI, AsyncTogether)):
        client.chat.completions.create = traced_create_async
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            client.responses.create = traced_response_create_async
    elif isinstance(client, AsyncAnthropic):
        client.messages.create = traced_create_async
        if original_stream:
            client.messages.stream = traced_stream_async
    elif isinstance(client, genai.client.AsyncClient):
        client.models.generate_content = traced_create_async
    elif isinstance(client, (OpenAI, Together)):
        client.chat.completions.create = traced_create_sync
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            client.responses.create = traced_response_create_sync
    elif isinstance(client, Anthropic):
        client.messages.create = traced_create_sync
        if original_stream:
            client.messages.stream = traced_stream_sync
    elif isinstance(client, genai.Client):
        client.models.generate_content = traced_create_sync
    
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
            - responses_method: Reference to the client's responses method (if applicable)
            - stream_method: Reference to the client's stream method (if applicable)
            
    Raises:
        ValueError: If client type is not supported
    """
    if isinstance(client, (OpenAI, AsyncOpenAI)):
        return "OPENAI_API_CALL", client.chat.completions.create, client.responses.create, None
    elif isinstance(client, (Together, AsyncTogether)):
        return "TOGETHER_API_CALL", client.chat.completions.create, None, None
    elif isinstance(client, (Anthropic, AsyncAnthropic)):
        return "ANTHROPIC_API_CALL", client.messages.create, None, client.messages.stream
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        return "GOOGLE_API_CALL", client.models.generate_content, None, None
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

def _format_response_output_data(client: ApiClient, response: Any) -> dict:
    """Format API response data based on client type.
    
    Normalizes different response formats into a consistent structure
    for tracing purposes.
    """
    message_content = None
    prompt_tokens = 0   
    completion_tokens = 0
    model_name = None
    if isinstance(client, (OpenAI, Together, AsyncOpenAI, AsyncTogether)):
        model_name = response.model
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        message_content = response.output
    else:
        warnings.warn(f"Unsupported client type: {type(client)}")
        return {}
    
    prompt_cost, completion_cost = cost_per_token(  
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    total_cost_usd = (prompt_cost + completion_cost) if prompt_cost and completion_cost else None
    usage = TraceUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_cost_usd=prompt_cost,
        completion_tokens_cost_usd=completion_cost,
        total_cost_usd=total_cost_usd,
        model_name=model_name
    )
    return message_content, usage


def _format_output_data(client: ApiClient, response: Any) -> dict:
    """Format API response data based on client type.
    
    Normalizes different response formats into a consistent structure
    for tracing purposes.
    
    Returns:
        dict containing:
            - content: The generated text
            - usage: Token usage statistics
    """
    prompt_tokens = 0
    completion_tokens = 0
    model_name = None
    message_content = None

    if isinstance(client, (OpenAI, Together, AsyncOpenAI, AsyncTogether)):
        model_name = response.model
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        message_content = response.choices[0].message.content
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        model_name = response.model_version
        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        message_content = response.candidates[0].content.parts[0].text
    elif isinstance(client, (Anthropic, AsyncAnthropic)):
        model_name = response.model
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        message_content = response.content[0].text
    else:
        warnings.warn(f"Unsupported client type: {type(client)}")
        return None, None
    
    prompt_cost, completion_cost = cost_per_token(
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    total_cost_usd = (prompt_cost + completion_cost) if prompt_cost and completion_cost else None
    usage = TraceUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_cost_usd=prompt_cost,
        completion_tokens_cost_usd=completion_cost,
        total_cost_usd=total_cost_usd,
        model_name=model_name
    )
    return message_content, usage

def combine_args_kwargs(func, args, kwargs):
    """
    Combine positional arguments and keyword arguments into a single dictionary.
    
    Args:
        func: The function being called
        args: Tuple of positional arguments
        kwargs: Dictionary of keyword arguments
        
    Returns:
        A dictionary combining both args and kwargs
    """
    try:
        import inspect
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        args_dict = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                args_dict[param_names[i]] = arg
            else:
                args_dict[f"arg{i}"] = arg
        
        return {**args_dict, **kwargs}
    except Exception as e:
        # Fallback if signature inspection fails
        return {**{f"arg{i}": arg for i, arg in enumerate(args)}, **kwargs}

# NOTE: This builds once, can be tweaked if we are missing / capturing other unncessary modules
# @link https://docs.python.org/3.13/library/sysconfig.html
_TRACE_FILEPATH_BLOCKLIST = tuple(
    os.path.realpath(p) + os.sep
    for p in {
        sysconfig.get_paths()['stdlib'],
        sysconfig.get_paths().get('platstdlib', ''),
        *site.getsitepackages(),
        site.getusersitepackages(),
        *(
            [os.path.join(os.path.dirname(__file__), '../../judgeval/')]
            if os.environ.get('JUDGMENT_DEV')
            else []
        ),
    } if p
)

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
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            # Check if usage is nested within choices (less common for final chunk, but check)
            elif chunk.choices and hasattr(chunk.choices[0], 'usage') and chunk.choices[0].usage:
                prompt_tokens = chunk.choices[0].usage.prompt_tokens
                completion_tokens = chunk.choices[0].usage.completion_tokens
                
            prompt_cost, completion_cost = cost_per_token(
                    model=chunk.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            total_cost_usd = (prompt_cost + completion_cost) if prompt_cost and completion_cost else None
            return TraceUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
                prompt_tokens_cost_usd=prompt_cost,
                completion_tokens_cost_usd=completion_cost,
                total_cost_usd=total_cost_usd,
                model_name=chunk.model
            )
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
    span: TraceSpan
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
        span.output = "".join(content_parts)
        span.usage = final_usage
        # Note: We might need to adjust _serialize_output if this dict causes issues,
        # but Pydantic's model_dump should handle dicts.

# --- Async Stream Wrapper ---
async def _async_stream_wrapper(
    original_stream: AsyncIterator,
    client: ApiClient,
    span: TraceSpan
) -> AsyncGenerator[Any, None]:
    # [Existing logic - unchanged]
    content_parts = []  # Use a list instead of string concatenation
    final_usage_data = None
    last_content_chunk = None
    anthropic_input_tokens = 0
    anthropic_output_tokens = 0

    target_span_id = span.span_id

    try:
        model_name = ""
        async for chunk in original_stream:
            # Check for OpenAI's final usage chunk
            if isinstance(client, (AsyncOpenAI, OpenAI)) and hasattr(chunk, 'usage') and chunk.usage is not None:
                final_usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens
                }
                model_name = chunk.model
                yield chunk
                continue

            if isinstance(client, (AsyncAnthropic, Anthropic)) and hasattr(chunk, 'type'):
                if chunk.type == "message_start":
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage') and hasattr(chunk.message.usage, 'input_tokens'):
                         anthropic_input_tokens = chunk.message.usage.input_tokens
                         model_name = chunk.message.model
                elif chunk.type == "message_delta":
                    if hasattr(chunk, 'usage') and hasattr(chunk.usage, 'output_tokens'):
                        anthropic_output_tokens = chunk.usage.output_tokens

            content_part = _extract_content_from_chunk(client, chunk)
            if content_part:
                content_parts.append(content_part)  # Append to list instead of concatenating
                last_content_chunk = chunk

            yield chunk
    finally:
        anthropic_final_usage = None
        if isinstance(client, (AsyncAnthropic, Anthropic)) and (anthropic_input_tokens > 0 or anthropic_output_tokens > 0):
             anthropic_final_usage = {
                 "prompt_tokens": anthropic_input_tokens,
                 "completion_tokens": anthropic_output_tokens,
                 "total_tokens": anthropic_input_tokens + anthropic_output_tokens
             }

        usage_info = None
        if final_usage_data:
             usage_info = final_usage_data
        elif anthropic_final_usage:
             usage_info = anthropic_final_usage
        elif last_content_chunk:
            usage_info = _extract_usage_from_final_chunk(client, last_content_chunk)

        if usage_info and not isinstance(usage_info, TraceUsage):
            prompt_cost, completion_cost = cost_per_token(  
                model=model_name,
                prompt_tokens=usage_info["prompt_tokens"],
                completion_tokens=usage_info["completion_tokens"],
            )
            usage_info = TraceUsage(
                prompt_tokens=usage_info["prompt_tokens"],
                completion_tokens=usage_info["completion_tokens"],
                total_tokens=usage_info["total_tokens"],
                prompt_tokens_cost_usd=prompt_cost,
                completion_tokens_cost_usd=completion_cost,
                total_cost_usd=prompt_cost + completion_cost,
                model_name=model_name
            )
        if span and hasattr(span, 'output'):
            span.output = ''.join(content_parts)
            span.usage = usage_info
            start_ts = getattr(span, 'created_at', time.time())
            span.duration = time.time() - start_ts
        # else: # Handle error case if necessary, but remove debug print

def cost_per_token(*args, **kwargs):
    try:
        return _original_cost_per_token(*args, **kwargs)
    except Exception as e:
        warnings.warn(f"Error calculating cost per token: {e}")
        return None, None

class _BaseStreamManagerWrapper:
    def __init__(self, original_manager, client, span_name, trace_client, stream_wrapper_func, input_kwargs):
        self._original_manager = original_manager
        self._client = client
        self._span_name = span_name
        self._trace_client = trace_client
        self._stream_wrapper_func = stream_wrapper_func
        self._input_kwargs = input_kwargs
        self._parent_span_id_at_entry = None

    def _create_span(self):
        start_time = time.time()
        span_id = str(uuid.uuid4())
        current_depth = 0
        if self._parent_span_id_at_entry and self._parent_span_id_at_entry in self._trace_client._span_depths:
            current_depth = self._trace_client._span_depths[self._parent_span_id_at_entry] + 1
        self._trace_client._span_depths[span_id] = current_depth
        span = TraceSpan(
            function=self._span_name,
            span_id=span_id,
            trace_id=self._trace_client.trace_id,
            depth=current_depth,
            message=self._span_name,
            created_at=start_time,
            span_type="llm",
            parent_span_id=self._parent_span_id_at_entry
        )
        self._trace_client.add_span(span)
        return span_id, span

    def _finalize_span(self, span_id):
        span = self._trace_client.span_id_to_span.get(span_id)
        if span:
            span.duration = time.time() - span.created_at
        if span_id in self._trace_client._span_depths:
            del self._trace_client._span_depths[span_id]

class _TracedAsyncStreamManagerWrapper(_BaseStreamManagerWrapper, AbstractAsyncContextManager):
    async def __aenter__(self):
        self._parent_span_id_at_entry = current_span_var.get()
        if not self._trace_client:
            return await self._original_manager.__aenter__()

        span_id, span = self._create_span()
        self._span_context_token = current_span_var.set(span_id)
        span.inputs = _format_input_data(self._client, **self._input_kwargs)

        # Call the original __aenter__ and expect it to be an async generator
        raw_iterator = await self._original_manager.__aenter__()
        span.output = "<pending stream>"
        return self._stream_wrapper_func(raw_iterator, self._client, span)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, '_span_context_token'):
            span_id = current_span_var.get()
            self._finalize_span(span_id)
            current_span_var.reset(self._span_context_token)
            delattr(self, '_span_context_token')
        return await self._original_manager.__aexit__(exc_type, exc_val, exc_tb)

class _TracedSyncStreamManagerWrapper(_BaseStreamManagerWrapper, AbstractContextManager):
    def __enter__(self):
        self._parent_span_id_at_entry = current_span_var.get()
        if not self._trace_client:
            return self._original_manager.__enter__()

        span_id, span = self._create_span()
        self._span_context_token = current_span_var.set(span_id)
        span.inputs = _format_input_data(self._client, **self._input_kwargs)

        raw_iterator = self._original_manager.__enter__()
        span.output = "<pending stream>"
        return self._stream_wrapper_func(raw_iterator, self._client, span)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, '_span_context_token'):
            span_id = current_span_var.get()
            self._finalize_span(span_id)
            current_span_var.reset(self._span_context_token)
            delattr(self, '_span_context_token')
        return self._original_manager.__exit__(exc_type, exc_val, exc_tb)

# --- Helper function for instance-prefixed qual_name ---
def get_instance_prefixed_name(instance, class_name, class_identifiers):
    """
    Returns the agent name (prefix) if the class and attribute are found in class_identifiers.
    Otherwise, returns None.
    """
    if class_name in class_identifiers:
        class_config = class_identifiers[class_name]
        attr = class_config['identifier']
            
        if hasattr(instance, attr):
            instance_name = getattr(instance, attr)
            return instance_name
        else:
            raise Exception(f"Attribute {attr} does not exist for {class_name}. Check your identify() decorator.")
    return None
