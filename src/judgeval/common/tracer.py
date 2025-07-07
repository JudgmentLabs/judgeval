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
from contextlib import (
    contextmanager,
    AbstractAsyncContextManager,
    AbstractContextManager,
)  # Import context manager bases
from dataclasses import dataclass
from datetime import datetime, timezone
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
from requests import RequestException
from judgeval.utils.requests import requests
from litellm import cost_per_token as _original_cost_per_token
from openai import OpenAI, AsyncOpenAI
from together import Together, AsyncTogether
from anthropic import Anthropic, AsyncAnthropic
from google import genai

# Local application/library-specific imports
from judgeval.constants import (
    JUDGMENT_TRACES_ADD_ANNOTATION_API_URL,
    JUDGMENT_TRACES_UPSERT_API_URL,
    JUDGMENT_TRACES_FETCH_API_URL,
    JUDGMENT_TRACES_DELETE_API_URL,
    JUDGMENT_PROJECT_DELETE_API_URL,
    JUDGMENT_TRACES_SPANS_BATCH_API_URL,
    JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL,
)
from judgeval.data import Example, Trace, TraceSpan, TraceUsage
from judgeval.scorers import APIScorerConfig, BaseScorer
from judgeval.evaluation_run import EvaluationRun
from judgeval.common.utils import ExcInfo, validate_api_key
from judgeval.common.exceptions import JudgmentAPIError

# Standard library imports needed for the new class
import concurrent.futures
from collections.abc import Iterator, AsyncIterator  # Add Iterator and AsyncIterator
import queue
import atexit

# Define context variables for tracking the current trace and the current span within a trace
current_trace_var = contextvars.ContextVar[Optional["TraceClient"]](
    "current_trace", default=None
)
current_span_var = contextvars.ContextVar[Optional[str]](
    "current_span", default=None
)  # ContextVar for the active span id

# Define type aliases for better code readability and maintainability
ApiClient: TypeAlias = Union[
    OpenAI,
    Together,
    Anthropic,
    AsyncOpenAI,
    AsyncAnthropic,
    AsyncTogether,
    genai.Client,
    genai.client.AsyncClient,
]  # Supported API clients
SpanType = Literal["span", "tool", "llm", "evaluation", "chain"]


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
            "annotation": {"text": self.text, "label": self.label, "score": self.score},
        }


class TraceManagerClient:
    """
    Client for handling trace endpoints with the Judgment API


    Operations include:
    - Fetching a trace by id
    - Saving a trace
    - Deleting a trace
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
        tracer: Optional["Tracer"] = None,
    ):
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
                "X-Organization-Id": self.organization_id,
            },
            verify=True,
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to fetch traces: {response.text}")

        return response.json()

    def upsert_trace(
        self,
        trace_data: dict,
        offline_mode: bool = False,
        show_link: bool = True,
        final_save: bool = True,
    ):
        """
        Upserts a trace to the Judgment API (always overwrites if exists).

        Args:
            trace_data: The trace data to upsert
            offline_mode: Whether running in offline mode
            show_link: Whether to show the UI link (for live tracing)
            final_save: Whether this is the final save (controls S3 saving)

        Returns:
            dict: Server response containing UI URL and other metadata
        """

        def fallback_encoder(obj):
            """
            Custom JSON encoder fallback.
            Tries to use obj.__repr__(), then str(obj) if that fails or for a simpler string.
            """
            try:
                return repr(obj)
            except Exception:
                try:
                    return str(obj)
                except Exception as e:
                    return f"<Unserializable object of type {type(obj).__name__}: {e}>"

        serialized_trace_data = json.dumps(trace_data, default=fallback_encoder)

        response = requests.post(
            JUDGMENT_TRACES_UPSERT_API_URL,
            data=serialized_trace_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id,
            },
            verify=True,
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to upsert trace data: {response.text}")

        # Parse server response
        server_response = response.json()

        # If S3 storage is enabled, save to S3 only on final save
        if self.tracer and self.tracer.use_s3 and final_save:
            try:
                s3_key = self.tracer.s3_storage.save_trace(
                    trace_data=trace_data,
                    trace_id=trace_data["trace_id"],
                    project_name=trace_data["project_name"],
                )
                print(f"Trace also saved to S3 at key: {s3_key}")
            except Exception as e:
                warnings.warn(f"Failed to save trace to S3: {str(e)}")

        if not offline_mode and show_link and "ui_results_url" in server_response:
            pretty_str = f"\n🔍 You can view your trace data here: [rgb(106,0,255)][link={server_response['ui_results_url']}]View Trace[/link]\n"
            rprint(pretty_str)

        return server_response

    ## TODO: Should have a log endpoint, endpoint should also support batched payloads
    def save_annotation(self, annotation: TraceAnnotation):
        json_data = {
            "span_id": annotation.span_id,
            "annotation": {
                "text": annotation.text,
                "label": annotation.label,
                "score": annotation.score,
            },
        }

        response = requests.post(
            JUDGMENT_TRACES_ADD_ANNOTATION_API_URL,
            json=json_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id,
            },
            verify=True,
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
                "X-Organization-Id": self.organization_id,
            },
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
                "X-Organization-Id": self.organization_id,
            },
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
                "X-Organization-Id": self.organization_id,
            },
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to delete traces: {response.text}")

        return response.json()


class TraceClient:
    """Client for managing a single trace context"""

    def __init__(
        self,
        tracer: "Tracer",
        trace_id: Optional[str] = None,
        name: str = "default",
        project_name: str | None = None,
        enable_monitoring: bool = True,
        enable_evaluations: bool = True,
        parent_trace_id: Optional[str] = None,
        parent_name: Optional[str] = None,
    ):
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.project_name = project_name or "default_project"
        self.tracer = tracer
        self.enable_monitoring = enable_monitoring
        self.enable_evaluations = enable_evaluations
        self.parent_trace_id = parent_trace_id
        self.parent_name = parent_name
        self.customer_id: Optional[str] = None  # Added customer_id attribute
        self.tags: List[Union[str, set, tuple]] = []  # Added tags attribute
        self.metadata: Dict[str, Any] = {}
        self.has_notification: Optional[bool] = False  # Initialize has_notification
        self.update_id: int = 1  # Initialize update_id to 1, increments with each save
        self.trace_spans: List[TraceSpan] = []
        self.span_id_to_span: Dict[str, TraceSpan] = {}
        self.evaluation_runs: List[EvaluationRun] = []
        self.annotations: List[TraceAnnotation] = []
        self.start_time: Optional[float] = (
            None  # Will be set after first successful save
        )
        self.trace_manager_client = TraceManagerClient(
            tracer.api_key, tracer.organization_id, tracer
        )
        self._span_depths: Dict[str, int] = {}  # NEW: To track depth of active spans

        # Get background span service from tracer
        self.background_span_service = (
            tracer.get_background_span_service() if tracer else None
        )

    def get_current_span(self):
        """Get the current span from the context var"""
        return self.tracer.get_current_span()

    def set_current_span(self, span: Any):
        """Set the current span from the context var"""
        return self.tracer.set_current_span(span)

    def reset_current_span(self, token: Any):
        """Reset the current span from the context var"""
        self.tracer.reset_current_span(token)

    @contextmanager
    def span(self, name: str, span_type: SpanType = "span"):
        """Context manager for creating a trace span, managing the current span via contextvars"""
        is_first_span = len(self.trace_spans) == 0
        if is_first_span:
            try:
                trace_id, server_response = self.save(final_save=False)
                # Set start_time after first successful save
                # Link will be shown by upsert_trace method
            except Exception as e:
                warnings.warn(f"Failed to save initial trace for live tracking: {e}")
        start_time = time.time()

        # Generate a unique ID for *this specific span invocation*
        span_id = str(uuid.uuid4())

        parent_span_id = (
            self.get_current_span()
        )  # Get ID of the parent span from context var
        token = self.set_current_span(
            span_id
        )  # Set *this* span's ID as the current one

        current_depth = 0
        if parent_span_id and parent_span_id in self._span_depths:
            current_depth = self._span_depths[parent_span_id] + 1

        self._span_depths[span_id] = current_depth  # Store depth by span_id

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

        # Queue span with initial state (input phase)
        if self.background_span_service:
            self.background_span_service.queue_span(span, span_state="input")

        try:
            yield self
        finally:
            duration = time.time() - start_time
            span.duration = duration

            # Queue span with completed state (output phase)
            if self.background_span_service:
                self.background_span_service.queue_span(span, span_state="completed")

            # Clean up depth tracking for this span_id
            if span_id in self._span_depths:
                del self._span_depths[span_id]
            # Reset context var
            self.reset_current_span(token)

    def async_evaluate(
        self,
        scorers: List[Union[APIScorerConfig, BaseScorer]],
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
        span_id: Optional[str] = None,  # <<< ADDED optional span_id parameter
    ):
        if not self.enable_evaluations:
            return

        start_time = time.time()  # Record start time

        try:
            # Load appropriate implementations for all scorers
            if not scorers:
                warnings.warn("No valid scorers available for evaluation")
                return

        except Exception as e:
            warnings.warn(f"Failed to load scorers: {str(e)}")
            return

        # If example is not provided, create one from the individual parameters
        if example is None:
            # Check if any of the individual parameters are provided
            if any(
                param is not None
                for param in [
                    input,
                    actual_output,
                    expected_output,
                    context,
                    retrieval_context,
                    tools_called,
                    expected_tools,
                    additional_metadata,
                ]
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
                )
            else:
                raise ValueError(
                    "Either 'example' or at least one of the individual parameters (input, actual_output, etc.) must be provided"
                )

        # Check examples before creating evaluation run

        # check_examples([example], scorers)

        # --- Modification: Capture span_id immediately ---
        # span_id_at_eval_call = current_span_var.get()
        # print(f"[TraceClient.async_evaluate] Captured span ID at eval call: {span_id_at_eval_call}")
        # Prioritize explicitly passed span_id, fallback to context var
        span_id_to_use = span_id if span_id is not None else self.get_current_span()
        # print(f"[TraceClient.async_evaluate] Using span_id: {span_id_to_use}")
        # --- End Modification ---

        eval_run = EvaluationRun(
            organization_id=self.tracer.organization_id,
            project_name=self.project_name,
            eval_name=f"{self.name.capitalize()}-"
            f"{span_id_to_use}-"  # Keep original eval name format using context var if available
            f"[{','.join(scorer.score_type.capitalize() for scorer in scorers)}]",
            examples=[example],
            scorers=scorers,
            model=model,
            judgment_api_key=self.tracer.api_key,
            trace_span_id=span_id_to_use,
        )

        self.add_eval_run(eval_run, start_time)  # Pass start_time to record_evaluation

        # Queue evaluation run through background service
        if self.background_span_service and span_id_to_use:
            # Get the current span data to avoid race conditions
            current_span = self.span_id_to_span.get(span_id_to_use)
            if current_span:
                self.background_span_service.queue_evaluation_run(
                    eval_run, span_id=span_id_to_use, span_data=current_span
                )

    def add_eval_run(self, eval_run: EvaluationRun, start_time: float):
        # --- Modification: Use span_id from eval_run ---
        current_span_id = eval_run.trace_span_id  # Get ID from the eval_run object
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
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            # Ignore self parameter
            if "self" in inputs:
                del inputs["self"]
            span.inputs = inputs

            # Queue span with input data
            try:
                if self.background_span_service:
                    self.background_span_service.queue_span(span, span_state="input")
            except Exception as e:
                warnings.warn(f"Failed to queue span with input data: {e}")

    def record_agent_name(self, agent_name: str):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.agent_name = agent_name

            # Queue span with agent_name data
            if self.background_span_service:
                self.background_span_service.queue_span(span, span_state="agent_name")

    def record_state_before(self, state: dict):
        """Records the agent's state before a tool execution on the current span.

        Args:
            state: A dictionary representing the agent's state.
        """
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.state_before = state

            # Queue span with state_before data
            if self.background_span_service:
                self.background_span_service.queue_span(span, span_state="state_before")

    def record_state_after(self, state: dict):
        """Records the agent's state after a tool execution on the current span.

        Args:
            state: A dictionary representing the agent's state.
        """
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.state_after = state

            # Queue span with state_after data
            if self.background_span_service:
                self.background_span_service.queue_span(span, span_state="state_after")

    async def _update_coroutine(self, span: TraceSpan, coroutine: Any, field: str):
        """Helper method to update the output of a trace entry once the coroutine completes"""
        try:
            result = await coroutine
            setattr(span, field, result)

            # Queue span with output data now that coroutine is complete
            if self.background_span_service and field == "output":
                self.background_span_service.queue_span(span, span_state="output")

            return result
        except Exception as e:
            setattr(span, field, f"Error: {str(e)}")

            # Queue span even if there was an error
            if self.background_span_service and field == "output":
                self.background_span_service.queue_span(span, span_state="output")

            raise

    def record_output(self, output: Any):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.output = "<pending>" if inspect.iscoroutine(output) else output

            if inspect.iscoroutine(output):
                asyncio.create_task(self._update_coroutine(span, output, "output"))

            # # Queue span with output data (unless it's pending)
            if self.background_span_service and not inspect.iscoroutine(output):
                self.background_span_service.queue_span(span, span_state="output")

            return span  # Return the created entry
        # Removed else block - original didn't have one
        return None  # Return None if no span_id found

    def record_usage(self, usage: TraceUsage):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.usage = usage

            # Queue span with usage data
            if self.background_span_service:
                self.background_span_service.queue_span(span, span_state="usage")

            return span  # Return the created entry
        # Removed else block - original didn't have one
        return None  # Return None if no span_id found

    def record_error(self, error: Dict[str, Any]):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.error = error

            # Queue span with error data
            if self.background_span_service:
                self.background_span_service.queue_span(span, span_state="error")

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
        if self.start_time is None:
            return 0.0  # No duration if trace hasn't been saved yet
        return time.time() - self.start_time

    def save(self, final_save: bool = False) -> Tuple[str, dict]:
        """
        Save the current trace to the database with rate limiting checks.
        First checks usage limits, then upserts the trace if allowed.

        Args:
            final_save: Whether this is the final save (updates usage counters)

        Returns a tuple of (trace_id, server_response) where server_response contains the UI URL and other metadata.
        """

        # Calculate total elapsed time
        total_duration = self.get_duration()

        # Create trace document
        trace_data = {
            "trace_id": self.trace_id,
            "name": self.name,
            "project_name": self.project_name,
            "created_at": datetime.fromtimestamp(
                self.start_time or time.time(), timezone.utc
            ).isoformat(),
            "duration": total_duration,
            "trace_spans": [span.model_dump() for span in self.trace_spans],
            "evaluation_runs": [run.model_dump() for run in self.evaluation_runs],
            "offline_mode": self.tracer.offline_mode,
            "parent_trace_id": self.parent_trace_id,
            "parent_name": self.parent_name,
            "customer_id": self.customer_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "update_id": self.update_id,
        }

        # If usage check passes, upsert the trace
        server_response = self.trace_manager_client.upsert_trace(
            trace_data,
            offline_mode=self.tracer.offline_mode,
            show_link=not final_save,  # Show link only on initial save, not final save
            final_save=final_save,  # Pass final_save to control S3 saving
        )

        if self.start_time is None:
            self.start_time = time.time()

        self.update_id += 1

        # Upload annotations
        # TODO: batch to the log endpoint
        for annotation in self.annotations:
            self.trace_manager_client.save_annotation(annotation)
        return self.trace_id, server_response

    def delete(self):
        return self.trace_manager_client.delete_trace(self.trace_id)

    def update_metadata(self, metadata: dict):
        """
        Set metadata for this trace.

        Args:
            metadata: Metadata as a dictionary

        Supported keys:
        - customer_id: ID of the customer using this trace
        - tags: List of tags for this trace
        - has_notification: Whether this trace has a notification
        - name: Name of the trace
        """
        for k, v in metadata.items():
            if k == "customer_id":
                if v is not None:
                    self.customer_id = str(v)
                else:
                    self.customer_id = None
            elif k == "tags":
                if isinstance(v, list):
                    # Validate that all items in the list are of the expected types
                    for item in v:
                        if not isinstance(item, (str, set, tuple)):
                            raise ValueError(
                                f"Tags must be a list of strings, sets, or tuples, got item of type {type(item)}"
                            )
                    self.tags = v
                else:
                    raise ValueError(
                        f"Tags must be a list of strings, sets, or tuples, got {type(v)}"
                    )
            elif k == "has_notification":
                if not isinstance(v, bool):
                    raise ValueError(
                        f"has_notification must be a boolean, got {type(v)}"
                    )
                self.has_notification = v
            elif k == "name":
                self.name = v
            else:
                self.metadata[k] = v

    def set_customer_id(self, customer_id: str):
        """
        Set the customer ID for this trace.

        Args:
            customer_id: The customer ID to set
        """
        self.update_metadata({"customer_id": customer_id})

    def set_tags(self, tags: List[Union[str, set, tuple]]):
        """
        Set the tags for this trace.

        Args:
            tags: List of tags to set
        """
        self.update_metadata({"tags": tags})


def _capture_exception_for_trace(
    current_trace: Optional["TraceClient"], exc_info: ExcInfo
):
    if not current_trace:
        return

    exc_type, exc_value, exc_traceback_obj = exc_info
    formatted_exception = {
        "type": exc_type.__name__ if exc_type else "UnknownExceptionType",
        "message": str(exc_value) if exc_value else "No exception message",
        "traceback": traceback.format_tb(exc_traceback_obj)
        if exc_traceback_obj
        else [],
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
    except Exception:
        pass

    current_trace.record_error(formatted_exception)

    # Queue the span with error state through background service
    if current_trace.background_span_service:
        current_span_id = current_trace.get_current_span()
        if current_span_id and current_span_id in current_trace.span_id_to_span:
            error_span = current_trace.span_id_to_span[current_span_id]
            current_trace.background_span_service.queue_span(
                error_span, span_state="error"
            )


class BackgroundSpanService:
    """
    Background service for queueing and batching trace spans for efficient saving.

    This service:
    - Queues spans as they complete
    - Batches them for efficient network usage
    - Sends spans periodically or when batches reach a certain size
    - Handles automatic flushing when the main event terminates
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        num_workers: int = 1,
    ):
        """
        Initialize the background span service.

        Args:
            judgment_api_key: API key for Judgment service
            organization_id: Organization ID
            batch_size: Number of spans to batch before sending (default: 10)
            flush_interval: Time in seconds between automatic flushes (default: 5.0)
            num_workers: Number of worker threads to process the queue (default: 1)
        """
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.num_workers = max(1, num_workers)  # Ensure at least 1 worker

        # Queue for pending spans
        self._span_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

        # Background threads for processing spans
        self._worker_threads: List[threading.Thread] = []
        self._shutdown_event = threading.Event()

        # Track spans that have been sent
        # self._sent_spans = set()

        # Register cleanup on exit
        atexit.register(self.shutdown)

        # Start the background workers
        self._start_workers()

    def _start_workers(self):
        """Start the background worker threads."""
        for i in range(self.num_workers):
            if len(self._worker_threads) < self.num_workers:
                worker_thread = threading.Thread(
                    target=self._worker_loop, daemon=True, name=f"SpanWorker-{i + 1}"
                )
                worker_thread.start()
                self._worker_threads.append(worker_thread)

    def _worker_loop(self):
        """Main worker loop that processes spans in batches."""
        batch = []
        last_flush_time = time.time()
        pending_task_count = (
            0  # Track how many tasks we've taken from queue but not marked done
        )

        while not self._shutdown_event.is_set() or self._span_queue.qsize() > 0:
            try:
                # First, do a blocking get to wait for at least one item
                if not batch:  # Only block if we don't have items already
                    try:
                        span_data = self._span_queue.get(timeout=1.0)
                        batch.append(span_data)
                        pending_task_count += 1
                    except queue.Empty:
                        # No new spans, continue to check for flush conditions
                        pass

                # Then, do non-blocking gets to drain any additional available items
                # up to our batch size limit
                while len(batch) < self.batch_size:
                    try:
                        span_data = self._span_queue.get_nowait()  # Non-blocking
                        batch.append(span_data)
                        pending_task_count += 1
                    except queue.Empty:
                        # No more items immediately available
                        break

                current_time = time.time()
                should_flush = len(batch) >= self.batch_size or (
                    batch and (current_time - last_flush_time) >= self.flush_interval
                )

                if should_flush and batch:
                    self._send_batch(batch)

                    # Only mark tasks as done after successful sending
                    for _ in range(pending_task_count):
                        self._span_queue.task_done()
                    pending_task_count = 0  # Reset counter

                    batch.clear()
                    last_flush_time = current_time

            except Exception as e:
                warnings.warn(f"Error in span service worker loop: {e}")
                # On error, still need to mark tasks as done to prevent hanging
                for _ in range(pending_task_count):
                    self._span_queue.task_done()
                pending_task_count = 0
                batch.clear()

        # Final flush on shutdown
        if batch:
            self._send_batch(batch)
            # Mark remaining tasks as done
            for _ in range(pending_task_count):
                self._span_queue.task_done()

    def _send_batch(self, batch: List[Dict[str, Any]]):
        """
        Send a batch of spans to the server.

        Args:
            batch: List of span dictionaries to send
        """
        if not batch:
            return

        try:
            # Group items by type for different endpoints
            spans_to_send = []
            evaluation_runs_to_send = []

            for item in batch:
                if item["type"] == "span":
                    spans_to_send.append(item["data"])
                elif item["type"] == "evaluation_run":
                    evaluation_runs_to_send.append(item["data"])

            # Send spans if any
            if spans_to_send:
                self._send_spans_batch(spans_to_send)

            # Send evaluation runs if any
            if evaluation_runs_to_send:
                self._send_evaluation_runs_batch(evaluation_runs_to_send)

        except Exception as e:
            warnings.warn(f"Failed to send batch: {e}")

    def _send_spans_batch(self, spans: List[Dict[str, Any]]):
        """Send a batch of spans to the spans endpoint."""
        payload = {"spans": spans, "organization_id": self.organization_id}

        # Serialize with fallback encoder
        def fallback_encoder(obj):
            try:
                return repr(obj)
            except Exception:
                try:
                    return str(obj)
                except Exception as e:
                    return f"<Unserializable object of type {type(obj).__name__}: {e}>"

        try:
            serialized_data = json.dumps(payload, default=fallback_encoder)

            # Send the actual HTTP request to the batch endpoint
            response = requests.post(
                JUDGMENT_TRACES_SPANS_BATCH_API_URL,
                data=serialized_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.judgment_api_key}",
                    "X-Organization-Id": self.organization_id,
                },
                verify=True,
                timeout=30,  # Add timeout to prevent hanging
            )

            if response.status_code != HTTPStatus.OK:
                warnings.warn(
                    f"Failed to send spans batch: HTTP {response.status_code} - {response.text}"
                )

        except RequestException as e:
            warnings.warn(f"Network error sending spans batch: {e}")
        except Exception as e:
            warnings.warn(f"Failed to serialize or send spans batch: {e}")

    def _send_evaluation_runs_batch(self, evaluation_runs: List[Dict[str, Any]]):
        """Send a batch of evaluation runs with their associated span data to the endpoint."""
        # Structure payload to include both evaluation run data and span data
        evaluation_entries = []
        for eval_data in evaluation_runs:
            # eval_data already contains the evaluation run data (no need to access ['data'])
            entry = {
                "evaluation_run": {
                    # Extract evaluation run fields (excluding span-specific fields)
                    key: value
                    for key, value in eval_data.items()
                    if key not in ["associated_span_id", "span_data", "queued_at"]
                },
                "associated_span": {
                    "span_id": eval_data.get("associated_span_id"),
                    "span_data": eval_data.get("span_data"),
                },
                "queued_at": eval_data.get("queued_at"),
            }
            evaluation_entries.append(entry)

        payload = {
            "organization_id": self.organization_id,
            "evaluation_entries": evaluation_entries,  # Each entry contains both eval run + span data
        }

        # Serialize with fallback encoder
        def fallback_encoder(obj):
            try:
                return repr(obj)
            except Exception:
                try:
                    return str(obj)
                except Exception as e:
                    return f"<Unserializable object of type {type(obj).__name__}: {e}>"

        try:
            serialized_data = json.dumps(payload, default=fallback_encoder)

            # Send the actual HTTP request to the batch endpoint
            response = requests.post(
                JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL,
                data=serialized_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.judgment_api_key}",
                    "X-Organization-Id": self.organization_id,
                },
                verify=True,
                timeout=30,  # Add timeout to prevent hanging
            )

            if response.status_code != HTTPStatus.OK:
                warnings.warn(
                    f"Failed to send evaluation runs batch: HTTP {response.status_code} - {response.text}"
                )

        except RequestException as e:
            warnings.warn(f"Network error sending evaluation runs batch: {e}")
        except Exception as e:
            warnings.warn(f"Failed to send evaluation runs batch: {e}")

    def queue_span(self, span: TraceSpan, span_state: str = "input"):
        """
        Queue a span for background sending.

        Args:
            span: The TraceSpan object to queue
            span_state: State of the span ("input", "output", "completed")
        """
        if not self._shutdown_event.is_set():
            span_data = {
                "type": "span",
                "data": {
                    **span.model_dump(),
                    "span_state": span_state,
                    "queued_at": time.time(),
                },
            }
            self._span_queue.put(span_data)

    def queue_evaluation_run(
        self, evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ):
        """
        Queue an evaluation run for background sending.

        Args:
            evaluation_run: The EvaluationRun object to queue
            span_id: The span ID associated with this evaluation run
            span_data: The span data at the time of evaluation (to avoid race conditions)
        """
        if not self._shutdown_event.is_set():
            eval_data = {
                "type": "evaluation_run",
                "data": {
                    **evaluation_run.model_dump(),
                    "associated_span_id": span_id,
                    "span_data": span_data.model_dump(),  # Include span data to avoid race conditions
                    "queued_at": time.time(),
                },
            }
            self._span_queue.put(eval_data)

    def flush(self):
        """Force immediate sending of all queued spans."""
        try:
            # Wait for the queue to be processed
            self._span_queue.join()
        except Exception as e:
            warnings.warn(f"Error during flush: {e}")

    def shutdown(self):
        """Shutdown the background service and flush remaining spans."""
        if self._shutdown_event.is_set():
            return

        try:
            # Signal shutdown to stop new items from being queued
            self._shutdown_event.set()

            # Try to flush any remaining spans
            try:
                self.flush()
            except Exception as e:
                warnings.warn(f"Error during final flush: {e}")
        except Exception as e:
            warnings.warn(f"Error during BackgroundSpanService shutdown: {e}")
        finally:
            # Clear the worker threads list (daemon threads will be killed automatically)
            self._worker_threads.clear()

    def get_queue_size(self) -> int:
        """Get the current size of the span queue."""
        return self._span_queue.qsize()


class _DeepTracer:
    _instance: Optional["_DeepTracer"] = None
    _lock: threading.Lock = threading.Lock()
    _refcount: int = 0
    _span_stack: contextvars.ContextVar[List[Dict[str, Any]]] = contextvars.ContextVar(
        "_deep_profiler_span_stack", default=[]
    )
    _skip_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
        "_deep_profiler_skip_stack", default=[]
    )
    _original_sys_trace: Optional[Callable] = None
    _original_threading_trace: Optional[Callable] = None

    def __init__(self, tracer: "Tracer"):
        self._tracer = tracer

    def _get_qual_name(self, frame) -> str:
        func_name = frame.f_code.co_name
        module_name = frame.f_globals.get("__name__", "unknown_module")

        try:
            func = frame.f_globals.get(func_name)
            if func is None:
                return f"{module_name}.{func_name}"
            if hasattr(func, "__qualname__"):
                return f"{module_name}.{func.__qualname__}"
            return f"{module_name}.{func_name}"
        except Exception:
            return f"{module_name}.{func_name}"

    def __new__(cls, tracer: "Tracer"):
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
        if func and (
            hasattr(func, "_judgment_span_name") or hasattr(func, "_judgment_span_type")
        ):
            return False

        if (
            not module_name
            or func_name.startswith("<")  # ex: <listcomp>
            or func_name.startswith("__")
            and func_name != "__call__"  # dunders
            or not self._is_user_code(frame.f_code.co_filename)
        ):
            return False

        return True

    @functools.cache
    def _is_user_code(self, filename: str):
        return (
            bool(filename)
            and not filename.startswith("<")
            and not os.path.realpath(filename).startswith(_TRACE_FILEPATH_BLOCKLIST)
        )

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

    def _cooperative_threading_trace(
        self, frame: types.FrameType, event: str, arg: Any
    ):
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

    def _trace(
        self, frame: types.FrameType, event: str, arg: Any, continuation_func: Callable
    ):
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False

        if not self._should_trace(frame):
            return

        if event not in ("call", "return", "exception"):
            return

        current_trace = self._tracer.get_current_trace()
        if not current_trace:
            return

        parent_span_id = self._tracer.get_current_span()
        if not parent_span_id:
            return

        qual_name = self._get_qual_name(frame)
        instance_name = None
        if "self" in frame.f_locals:
            instance = frame.f_locals["self"]
            class_name = instance.__class__.__name__
            class_identifiers = getattr(self._tracer, "class_identifiers", {})
            instance_name = get_instance_prefixed_name(
                instance, class_name, class_identifiers
            )
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

            span_stack.append(
                {
                    "span_id": span_id,
                    "parent_span_id": parent_span_id,
                    "function": qual_name,
                    "start_time": start_time,
                }
            )
            self._span_stack.set(span_stack)

            token = self._tracer.set_current_span(span_id)
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
                agent_name=instance_name,
            )
            current_trace.add_span(span)

            inputs = {}
            try:
                args_info = inspect.getargvalues(frame)
                for arg in args_info.args:
                    try:
                        inputs[arg] = args_info.locals.get(arg)
                    except Exception:
                        inputs[arg] = "<<Unserializable>>"
                current_trace.record_input(inputs)
            except Exception as e:
                current_trace.record_input({"error": str(e)})

        elif event == "return":
            if not span_stack:
                return

            current_id = self._tracer.get_current_span()

            span_data = None
            for i, entry in enumerate(reversed(span_stack)):
                if entry["span_id"] == current_id:
                    span_data = span_stack.pop(-(i + 1))
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
                self._tracer.set_current_span(span_stack[-1]["span_id"])
            else:
                self._tracer.set_current_span(span_data["parent_span_id"])

            if "_judgment_span_token" in frame.f_locals:
                self._tracer.reset_current_span(frame.f_locals["_judgment_span_token"])

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


# Below commented out function isn't used anymore?

# def log(self, message: str, level: str = "info"):
#         """ Log a message with the span context """
#         current_trace = self._tracer.get_current_trace()
#         if current_trace:
#             current_trace.log(message, level)
#         else:
#             print(f"[{level}] {message}")
#         current_trace.record_output({"log": message})


class Tracer:
    # Tracer.current_trace class variable is currently used in wrap()
    # TODO: Keep track of cross-context state for current trace and current span ID solely through class variables instead of instance variables?
    # Should be fine to do so as long as we keep Tracer as a singleton
    current_trace: Optional[TraceClient] = None
    # current_span_id: Optional[str] = None

    trace_across_async_contexts: bool = (
        False  # BY default, we don't trace across async contexts
    )

    def __init__(
        self,
        api_key: str | None = os.getenv("JUDGMENT_API_KEY"),
        organization_id: str | None = os.getenv("JUDGMENT_ORG_ID"),
        project_name: str | None = None,
        deep_tracing: bool = False,  # Deep tracing is disabled by default
        enable_monitoring: bool = os.getenv("JUDGMENT_MONITORING", "true").lower()
        == "true",
        enable_evaluations: bool = os.getenv("JUDGMENT_EVALUATIONS", "true").lower()
        == "true",
        # S3 configuration
        use_s3: bool = False,
        s3_bucket_name: Optional[str] = None,
        s3_aws_access_key_id: Optional[str] = None,
        s3_aws_secret_access_key: Optional[str] = None,
        s3_region_name: Optional[str] = None,
        trace_across_async_contexts: bool = False,  # BY default, we don't trace across async contexts
        # Background span service configuration
        span_batch_size: int = 50,  # Number of spans to batch before sending
        span_flush_interval: float = 1.0,  # Time in seconds between automatic flushes
        span_num_workers: int = 10,  # Number of worker threads for span processing
    ):
        if not api_key:
            raise ValueError("Tracer must be configured with a Judgment API key")

        try:
            result, response = validate_api_key(api_key)
        except Exception as e:
            print(f"Issue with verifying API key, disabling monitoring: {e}")
            enable_monitoring = False
            result = True

        if not result:
            raise JudgmentAPIError(f"Issue with passed in Judgment API key: {response}")

        if not organization_id:
            raise ValueError("Tracer must be configured with an Organization ID")
        if use_s3 and not s3_bucket_name:
            raise ValueError("S3 bucket name must be provided when use_s3 is True")

        self.api_key: str = api_key
        self.project_name: str = project_name or "default_project"
        self.organization_id: str = organization_id
        self.traces: List[Trace] = []
        self.enable_monitoring: bool = enable_monitoring
        self.enable_evaluations: bool = enable_evaluations
        self.class_identifiers: Dict[
            str, str
        ] = {}  # Dictionary to store class identifiers
        self.span_id_to_previous_span_id: Dict[str, str | None] = {}
        self.trace_id_to_previous_trace: Dict[str, TraceClient | None] = {}
        self.current_span_id: Optional[str] = None
        self.current_trace: Optional[TraceClient] = None
        self.trace_across_async_contexts: bool = trace_across_async_contexts
        Tracer.trace_across_async_contexts = trace_across_async_contexts

        # Initialize S3 storage if enabled
        self.use_s3 = use_s3
        if use_s3:
            from judgeval.common.s3_storage import S3Storage

            try:
                self.s3_storage = S3Storage(
                    bucket_name=s3_bucket_name,
                    aws_access_key_id=s3_aws_access_key_id,
                    aws_secret_access_key=s3_aws_secret_access_key,
                    region_name=s3_region_name,
                )
            except Exception as e:
                print(f"Issue with initializing S3 storage, disabling S3: {e}")
                self.use_s3 = False

        self.offline_mode = False  # This is used to differentiate traces between online and offline (IE experiments vs monitoring page)
        self.deep_tracing: bool = deep_tracing  # NEW: Store deep tracing setting

        # Initialize background span service
        self.background_span_service: Optional[BackgroundSpanService] = None
        self.background_span_service = BackgroundSpanService(
            judgment_api_key=api_key,
            organization_id=organization_id,
            batch_size=span_batch_size,
            flush_interval=span_flush_interval,
            num_workers=span_num_workers,
        )

    def set_current_span(self, span_id: str) -> Optional[contextvars.Token[str | None]]:
        self.span_id_to_previous_span_id[span_id] = self.current_span_id
        self.current_span_id = span_id
        Tracer.current_span_id = span_id
        try:
            token = current_span_var.set(span_id)
        except Exception:
            token = None
        return token

    def get_current_span(self) -> Optional[str]:
        try:
            current_span_var_val = current_span_var.get()
        except Exception:
            current_span_var_val = None
        return (
            (self.current_span_id or current_span_var_val)
            if self.trace_across_async_contexts
            else current_span_var_val
        )

    def reset_current_span(
        self,
        token: Optional[contextvars.Token[str | None]] = None,
        span_id: Optional[str] = None,
    ):
        try:
            if token:
                current_span_var.reset(token)
        except Exception:
            pass
        if not span_id:
            span_id = self.current_span_id
        if span_id:
            self.current_span_id = self.span_id_to_previous_span_id.get(span_id)
            Tracer.current_span_id = self.current_span_id

    def set_current_trace(
        self, trace: TraceClient
    ) -> Optional[contextvars.Token[TraceClient | None]]:
        """
        Set the current trace context in contextvars
        """
        self.trace_id_to_previous_trace[trace.trace_id] = self.current_trace
        self.current_trace = trace
        Tracer.current_trace = trace
        try:
            token = current_trace_var.set(trace)
        except Exception:
            token = None
        return token

    def get_current_trace(self) -> Optional[TraceClient]:
        """
        Get the current trace context.

        Tries to get the trace client from the context variable first.
        If not found (e.g., context lost across threads/tasks),
        it falls back to the active trace client managed by the callback handler.
        """
        try:
            current_trace_var_val = current_trace_var.get()
        except Exception:
            current_trace_var_val = None
        return (
            (self.current_trace or current_trace_var_val)
            if self.trace_across_async_contexts
            else current_trace_var_val
        )

    def reset_current_trace(
        self,
        token: Optional[contextvars.Token[TraceClient | None]] = None,
        trace_id: Optional[str] = None,
    ):
        try:
            if token:
                current_trace_var.reset(token)
        except Exception:
            pass
        if not trace_id and self.current_trace:
            trace_id = self.current_trace.trace_id
        if trace_id:
            self.current_trace = self.trace_id_to_previous_trace.get(trace_id)
            Tracer.current_trace = self.current_trace

    @contextmanager
    def trace(
        self, name: str, project_name: str | None = None
    ) -> Generator[TraceClient, None, None]:
        """Start a new trace context using a context manager"""
        trace_id = str(uuid.uuid4())
        project = project_name if project_name is not None else self.project_name

        # Get parent trace info from context
        parent_trace = self.get_current_trace()
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
            enable_monitoring=self.enable_monitoring,
            enable_evaluations=self.enable_evaluations,
            parent_trace_id=parent_trace_id,
            parent_name=parent_name,
        )

        # Set the current trace in context variables
        token = self.set_current_trace(trace)

        # Automatically create top-level span
        with trace.span(name or "unnamed_trace"):
            try:
                # Save the trace to the database to handle Evaluations' trace_id referential integrity
                yield trace
            finally:
                # Reset the context variable
                self.reset_current_trace(token)

    def log(self, msg: str, label: str = "log", score: int = 1):
        """Log a message with the current span context"""
        current_span_id = self.get_current_span()
        current_trace = self.get_current_trace()
        if current_span_id and current_trace:
            annotation = TraceAnnotation(
                span_id=current_span_id, text=msg, label=label, score=score
            )
            current_trace.add_annotation(annotation)

        rprint(f"[bold]{label}:[/bold] {msg}")

    def identify(
        self,
        identifier: str,
        track_state: bool = False,
        track_attributes: Optional[List[str]] = None,
        field_mappings: Optional[Dict[str, str]] = None,
    ):
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
                "field_mappings": field_mappings or {},
            }
            return cls

        return decorator

    def _capture_instance_state(
        self, instance: Any, class_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Capture the state of an instance based on class configuration.
        Args:
            instance: The instance to capture the state of.
            class_config: Configuration dictionary for state capture,
                          expected to contain 'track_attributes' and 'field_mappings'.
        """
        track_attributes = class_config.get("track_attributes")
        field_mappings = class_config.get("field_mappings")

        if track_attributes:
            state = {attr: getattr(instance, attr, None) for attr in track_attributes}
        else:
            state = {
                k: v for k, v in instance.__dict__.items() if not k.startswith("_")
            }

        if field_mappings:
            state["field_mappings"] = field_mappings

        return state

    def _get_instance_state_if_tracked(self, args):
        """
        Extract instance state if the instance should be tracked.

        Returns the captured state dict if tracking is enabled, None otherwise.
        """
        if args and hasattr(args[0], "__class__"):
            instance = args[0]
            class_name = instance.__class__.__name__
            if (
                class_name in self.class_identifiers
                and isinstance(self.class_identifiers[class_name], dict)
                and self.class_identifiers[class_name].get("track_state", False)
            ):
                return self._capture_instance_state(
                    instance, self.class_identifiers[class_name]
                )

    def _conditionally_capture_and_record_state(
        self, trace_client_instance: TraceClient, args: tuple, is_before: bool
    ):
        """Captures instance state if tracked and records it via the trace_client."""
        state = self._get_instance_state_if_tracked(args)
        if state:
            if is_before:
                trace_client_instance.record_state_before(state)
            else:
                trace_client_instance.record_state_after(state)

    def observe(
        self,
        func=None,
        *,
        name=None,
        span_type: SpanType = "span",
    ):
        """
        Decorator to trace function execution with detailed entry/exit information.

        Args:
            func: The function to decorate
            name: Optional custom name for the span (defaults to function name)
            span_type: Type of span (default "span").
        """
        # If monitoring is disabled, return the function as is
        try:
            if not self.enable_monitoring:
                return func if func else lambda f: f

            if func is None:
                return lambda f: self.observe(
                    f,
                    name=name,
                    span_type=span_type,
                )

            # Use provided name or fall back to function name
            original_span_name = name or func.__name__

            # Store custom attributes on the function object
            func._judgment_span_name = original_span_name
            func._judgment_span_type = span_type

        except Exception:
            return func

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                nonlocal original_span_name
                class_name = None
                span_name = original_span_name
                agent_name = None

                if args and hasattr(args[0], "__class__"):
                    class_name = args[0].__class__.__name__
                    agent_name = get_instance_prefixed_name(
                        args[0], class_name, self.class_identifiers
                    )

                # Get current trace from context
                current_trace = self.get_current_trace()

                # If there's no current trace, create a root trace
                if not current_trace:
                    trace_id = str(uuid.uuid4())
                    project = self.project_name

                    # Create a new trace client to serve as the root
                    current_trace = TraceClient(
                        self,
                        trace_id,
                        span_name,  # MODIFIED: Use span_name directly
                        project_name=project,
                        enable_monitoring=self.enable_monitoring,
                        enable_evaluations=self.enable_evaluations,
                    )

                    trace_token = self.set_current_trace(current_trace)

                    try:
                        # Use span for the function execution within the root trace
                        # This sets the current_span_var
                        with current_trace.span(
                            span_name, span_type=span_type
                        ) as span:  # MODIFIED: Use span_name directly
                            # Record inputs
                            inputs = combine_args_kwargs(func, args, kwargs)
                            span.record_input(inputs)
                            if agent_name:
                                span.record_agent_name(agent_name)

                            # Capture state before execution
                            self._conditionally_capture_and_record_state(
                                span, args, is_before=True
                            )

                            try:
                                if self.deep_tracing:
                                    with _DeepTracer(self):
                                        result = await func(*args, **kwargs)
                                else:
                                    result = await func(*args, **kwargs)
                            except Exception as e:
                                _capture_exception_for_trace(
                                    current_trace, sys.exc_info()
                                )
                                raise e

                            # Capture state after execution
                            self._conditionally_capture_and_record_state(
                                span, args, is_before=False
                            )

                            # Record output
                            span.record_output(result)
                        return result
                    finally:
                        # Flush background spans before saving the trace
                        try:
                            complete_trace_data = {
                                "trace_id": current_trace.trace_id,
                                "name": current_trace.name,
                                "created_at": datetime.fromtimestamp(
                                    current_trace.start_time or time.time(),
                                    timezone.utc,
                                ).isoformat(),
                                "duration": current_trace.get_duration(),
                                "trace_spans": [
                                    span.model_dump()
                                    for span in current_trace.trace_spans
                                ],
                                "offline_mode": self.offline_mode,
                                "parent_trace_id": current_trace.parent_trace_id,
                                "parent_name": current_trace.parent_name,
                            }
                            # Save the completed trace
                            trace_id, server_response = current_trace.save(
                                final_save=True
                            )

                            # Store the complete trace data instead of just server response

                            self.traces.append(complete_trace_data)

                            # if self.background_span_service:
                            #     self.background_span_service.flush()

                            # Reset trace context (span context resets automatically)
                            self.reset_current_trace(trace_token)
                        except Exception as e:
                            warnings.warn(f"Issue with async_wrapper: {e}")
                            return
                else:
                    with current_trace.span(span_name, span_type=span_type) as span:
                        inputs = combine_args_kwargs(func, args, kwargs)
                        span.record_input(inputs)
                        if agent_name:
                            span.record_agent_name(agent_name)

                        # Capture state before execution
                        self._conditionally_capture_and_record_state(
                            span, args, is_before=True
                        )

                        try:
                            if self.deep_tracing:
                                with _DeepTracer(self):
                                    result = await func(*args, **kwargs)
                            else:
                                result = await func(*args, **kwargs)
                        except Exception as e:
                            _capture_exception_for_trace(current_trace, sys.exc_info())
                            raise e

                        # Capture state after execution
                        self._conditionally_capture_and_record_state(
                            span, args, is_before=False
                        )

                        span.record_output(result)
                    return result

            return async_wrapper
        else:
            # Non-async function implementation with deep tracing
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal original_span_name
                class_name = None
                span_name = original_span_name
                agent_name = None
                if args and hasattr(args[0], "__class__"):
                    class_name = args[0].__class__.__name__
                    agent_name = get_instance_prefixed_name(
                        args[0], class_name, self.class_identifiers
                    )
                # Get current trace from context
                current_trace = self.get_current_trace()

                # If there's no current trace, create a root trace
                if not current_trace:
                    trace_id = str(uuid.uuid4())
                    project = self.project_name

                    # Create a new trace client to serve as the root
                    current_trace = TraceClient(
                        self,
                        trace_id,
                        span_name,  # MODIFIED: Use span_name directly
                        project_name=project,
                        enable_monitoring=self.enable_monitoring,
                        enable_evaluations=self.enable_evaluations,
                    )

                    trace_token = self.set_current_trace(current_trace)

                    try:
                        # Use span for the function execution within the root trace
                        # This sets the current_span_var
                        with current_trace.span(
                            span_name, span_type=span_type
                        ) as span:  # MODIFIED: Use span_name directly
                            # Record inputs
                            inputs = combine_args_kwargs(func, args, kwargs)
                            span.record_input(inputs)
                            if agent_name:
                                span.record_agent_name(agent_name)
                            # Capture state before execution
                            self._conditionally_capture_and_record_state(
                                span, args, is_before=True
                            )

                            try:
                                if self.deep_tracing:
                                    with _DeepTracer(self):
                                        result = func(*args, **kwargs)
                                else:
                                    result = func(*args, **kwargs)
                            except Exception as e:
                                _capture_exception_for_trace(
                                    current_trace, sys.exc_info()
                                )
                                raise e

                            # Capture state after execution
                            self._conditionally_capture_and_record_state(
                                span, args, is_before=False
                            )

                            # Record output
                            span.record_output(result)
                        return result
                    finally:
                        # Flush background spans before saving the trace
                        try:
                            # Save the completed trace
                            trace_id, server_response = current_trace.save(
                                final_save=True
                            )

                            # Store the complete trace data instead of just server response
                            complete_trace_data = {
                                "trace_id": current_trace.trace_id,
                                "name": current_trace.name,
                                "created_at": datetime.fromtimestamp(
                                    current_trace.start_time or time.time(),
                                    timezone.utc,
                                ).isoformat(),
                                "duration": current_trace.get_duration(),
                                "trace_spans": [
                                    span.model_dump()
                                    for span in current_trace.trace_spans
                                ],
                                "offline_mode": self.offline_mode,
                                "parent_trace_id": current_trace.parent_trace_id,
                                "parent_name": current_trace.parent_name,
                            }
                            self.traces.append(complete_trace_data)
                            # Reset trace context (span context resets automatically)
                            self.reset_current_trace(trace_token)
                        except Exception as e:
                            warnings.warn(f"Issue with save: {e}")
                            return
                else:
                    with current_trace.span(span_name, span_type=span_type) as span:
                        inputs = combine_args_kwargs(func, args, kwargs)
                        span.record_input(inputs)
                        if agent_name:
                            span.record_agent_name(agent_name)

                        # Capture state before execution
                        self._conditionally_capture_and_record_state(
                            span, args, is_before=True
                        )

                        try:
                            if self.deep_tracing:
                                with _DeepTracer(self):
                                    result = func(*args, **kwargs)
                            else:
                                result = func(*args, **kwargs)
                        except Exception as e:
                            _capture_exception_for_trace(current_trace, sys.exc_info())
                            raise e

                        # Capture state after execution
                        self._conditionally_capture_and_record_state(
                            span, args, is_before=False
                        )

                        span.record_output(result)
                    return result

            return wrapper

    def observe_tools(
        self,
        cls=None,
        *,
        exclude_methods: Optional[List[str]] = None,
        include_private: bool = False,
        warn_on_double_decoration: bool = True,
    ):
        """
        Automatically adds @observe(span_type="tool") to all methods in a class.

        Args:
            cls: The class to decorate (automatically provided when used as decorator)
            exclude_methods: List of method names to skip decorating. Defaults to common magic methods
            include_private: Whether to decorate methods starting with underscore. Defaults to False
            warn_on_double_decoration: Whether to print warnings when skipping already-decorated methods. Defaults to True
        """

        if exclude_methods is None:
            exclude_methods = ["__init__", "__new__", "__del__", "__str__", "__repr__"]

        def decorate_class(cls):
            if not self.enable_monitoring:
                return cls

            decorated = []
            skipped = []

            for name in dir(cls):
                method = getattr(cls, name)

                if (
                    not callable(method)
                    or name in exclude_methods
                    or (name.startswith("_") and not include_private)
                    or not hasattr(cls, name)
                ):
                    continue

                if hasattr(method, "_judgment_span_name"):
                    skipped.append(name)
                    if warn_on_double_decoration:
                        print(
                            f"Warning: {cls.__name__}.{name} already decorated, skipping"
                        )
                    continue

                try:
                    decorated_method = self.observe(method, span_type="tool")
                    setattr(cls, name, decorated_method)
                    decorated.append(name)
                except Exception as e:
                    if warn_on_double_decoration:
                        print(f"Warning: Failed to decorate {cls.__name__}.{name}: {e}")

            return cls

        return decorate_class if cls is None else decorate_class(cls)

    def async_evaluate(self, *args, **kwargs):
        try:
            if not self.enable_monitoring or not self.enable_evaluations:
                return

            # --- Get trace_id passed explicitly (if any) ---
            passed_trace_id = kwargs.pop(
                "trace_id", None
            )  # Get and remove trace_id from kwargs

            current_trace = self.get_current_trace()

            if current_trace:
                # Pass the explicitly provided trace_id if it exists, otherwise let async_evaluate handle it
                # (Note: TraceClient.async_evaluate doesn't currently use an explicit trace_id, but this is for future proofing/consistency)
                if passed_trace_id:
                    kwargs["trace_id"] = (
                        passed_trace_id  # Re-add if needed by TraceClient.async_evaluate
                    )
                current_trace.async_evaluate(*args, **kwargs)
            else:
                warnings.warn(
                    "No trace found (context var or fallback), skipping evaluation"
                )  # Modified warning
        except Exception as e:
            warnings.warn(f"Issue with async_evaluate: {e}")

    def update_metadata(self, metadata: dict):
        """
        Update metadata for the current trace.

        Args:
            metadata: Metadata as a dictionary
        """
        current_trace = self.get_current_trace()
        if current_trace:
            current_trace.update_metadata(metadata)
        else:
            warnings.warn("No current trace found, cannot set metadata")

    def set_customer_id(self, customer_id: str):
        """
        Set the customer ID for the current trace.

        Args:
            customer_id: The customer ID to set
        """
        current_trace = self.get_current_trace()
        if current_trace:
            current_trace.set_customer_id(customer_id)
        else:
            warnings.warn("No current trace found, cannot set customer ID")

    def set_tags(self, tags: List[Union[str, set, tuple]]):
        """
        Set the tags for the current trace.

        Args:
            tags: List of tags to set
        """
        current_trace = self.get_current_trace()
        if current_trace:
            current_trace.set_tags(tags)
        else:
            warnings.warn("No current trace found, cannot set tags")

    def get_background_span_service(self) -> Optional[BackgroundSpanService]:
        """Get the background span service instance."""
        return self.background_span_service

    def flush_background_spans(self):
        """Flush all pending spans in the background service."""
        if self.background_span_service:
            self.background_span_service.flush()

    def shutdown_background_service(self):
        """Shutdown the background span service."""
        if self.background_span_service:
            self.background_span_service.shutdown()
            self.background_span_service = None


def _get_current_trace(
    trace_across_async_contexts: bool = Tracer.trace_across_async_contexts,
):
    if trace_across_async_contexts:
        return Tracer.current_trace
    else:
        return current_trace_var.get()


def wrap(
    client: Any, trace_across_async_contexts: bool = Tracer.trace_across_async_contexts
) -> Any:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, and Google GenAI clients.
    Patches both '.create' and Anthropic's '.stream' methods using a wrapper class.
    """
    (
        span_name,
        original_create,
        original_responses_create,
        original_stream,
        original_beta_parse,
    ) = _get_client_config(client)

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
                    UserWarning,
                )

        return is_streaming

    def _format_and_record_output(span, response, is_streaming, is_async, is_responses):
        """Format and record the output in the span"""
        if is_streaming:
            output_entry = span.record_output("<pending stream>")
            wrapper_func = _async_stream_wrapper if is_async else _sync_stream_wrapper
            return wrapper_func(
                response, client, output_entry, trace_across_async_contexts
            )
        else:
            format_func = (
                _format_response_output_data if is_responses else _format_output_data
            )
            output, usage = format_func(client, response)
            span.record_output(output)
            span.record_usage(usage)

            # Queue the completed LLM span now that it has all data (input, output, usage)
            current_trace = _get_current_trace(trace_across_async_contexts)
            if current_trace and current_trace.background_span_service:
                # Get the current span from the trace client
                current_span_id = current_trace.get_current_span()
                if current_span_id and current_span_id in current_trace.span_id_to_span:
                    completed_span = current_trace.span_id_to_span[current_span_id]
                    current_trace.background_span_service.queue_span(
                        completed_span, span_state="completed"
                    )

            return response

    # --- Traced Async Functions ---
    async def traced_create_async(*args, **kwargs):
        current_trace = _get_current_trace(trace_across_async_contexts)
        if not current_trace:
            return await original_create(*args, **kwargs)

        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(span, kwargs)

            try:
                response_or_iterator = await original_create(*args, **kwargs)
                return _format_and_record_output(
                    span, response_or_iterator, is_streaming, True, False
                )
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e

    async def traced_beta_parse_async(*args, **kwargs):
        current_trace = _get_current_trace(trace_across_async_contexts)
        if not current_trace:
            return await original_beta_parse(*args, **kwargs)

        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(span, kwargs)

            try:
                response_or_iterator = await original_beta_parse(*args, **kwargs)
                return _format_and_record_output(
                    span, response_or_iterator, is_streaming, True, False
                )
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e

    # Async responses for OpenAI clients
    async def traced_response_create_async(*args, **kwargs):
        current_trace = _get_current_trace(trace_across_async_contexts)
        if not current_trace:
            return await original_responses_create(*args, **kwargs)

        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(
                span, kwargs, is_responses=True
            )

            try:
                response_or_iterator = await original_responses_create(*args, **kwargs)
                return _format_and_record_output(
                    span, response_or_iterator, is_streaming, True, True
                )
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e

    # Function replacing .stream() for async clients
    def traced_stream_async(*args, **kwargs):
        current_trace = _get_current_trace(trace_across_async_contexts)
        if not current_trace or not original_stream:
            return original_stream(*args, **kwargs)

        original_manager = original_stream(*args, **kwargs)
        return _TracedAsyncStreamManagerWrapper(
            original_manager=original_manager,
            client=client,
            span_name=span_name,
            trace_client=current_trace,
            stream_wrapper_func=_async_stream_wrapper,
            input_kwargs=kwargs,
            trace_across_async_contexts=trace_across_async_contexts,
        )

    # --- Traced Sync Functions ---
    def traced_create_sync(*args, **kwargs):
        current_trace = _get_current_trace(trace_across_async_contexts)
        if not current_trace:
            return original_create(*args, **kwargs)

        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(span, kwargs)

            try:
                response_or_iterator = original_create(*args, **kwargs)
                return _format_and_record_output(
                    span, response_or_iterator, is_streaming, False, False
                )
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e

    def traced_beta_parse_sync(*args, **kwargs):
        current_trace = _get_current_trace(trace_across_async_contexts)
        if not current_trace:
            return original_beta_parse(*args, **kwargs)

        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(span, kwargs)

            try:
                response_or_iterator = original_beta_parse(*args, **kwargs)
                return _format_and_record_output(
                    span, response_or_iterator, is_streaming, False, False
                )
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e

    def traced_response_create_sync(*args, **kwargs):
        current_trace = _get_current_trace(trace_across_async_contexts)
        if not current_trace:
            return original_responses_create(*args, **kwargs)

        with current_trace.span(span_name, span_type="llm") as span:
            is_streaming = _record_input_and_check_streaming(
                span, kwargs, is_responses=True
            )

            try:
                response_or_iterator = original_responses_create(*args, **kwargs)
                return _format_and_record_output(
                    span, response_or_iterator, is_streaming, False, True
                )
            except Exception as e:
                _capture_exception_for_trace(span, sys.exc_info())
                raise e

    # Function replacing sync .stream()
    def traced_stream_sync(*args, **kwargs):
        current_trace = _get_current_trace(trace_across_async_contexts)
        if not current_trace or not original_stream:
            return original_stream(*args, **kwargs)

        original_manager = original_stream(*args, **kwargs)
        return _TracedSyncStreamManagerWrapper(
            original_manager=original_manager,
            client=client,
            span_name=span_name,
            trace_client=current_trace,
            stream_wrapper_func=_sync_stream_wrapper,
            input_kwargs=kwargs,
            trace_across_async_contexts=trace_across_async_contexts,
        )

    # --- Assign Traced Methods to Client Instance ---
    if isinstance(client, (AsyncOpenAI, AsyncTogether)):
        client.chat.completions.create = traced_create_async
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            client.responses.create = traced_response_create_async
        if (
            hasattr(client, "beta")
            and hasattr(client.beta, "chat")
            and hasattr(client.beta.chat, "completions")
            and hasattr(client.beta.chat.completions, "parse")
        ):
            client.beta.chat.completions.parse = traced_beta_parse_async
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
        if (
            hasattr(client, "beta")
            and hasattr(client.beta, "chat")
            and hasattr(client.beta.chat, "completions")
            and hasattr(client.beta.chat.completions, "parse")
        ):
            client.beta.chat.completions.parse = traced_beta_parse_sync
    elif isinstance(client, Anthropic):
        client.messages.create = traced_create_sync
        if original_stream:
            client.messages.stream = traced_stream_sync
    elif isinstance(client, genai.Client):
        client.models.generate_content = traced_create_sync

    return client


# Helper functions for client-specific operations


def _get_client_config(
    client: ApiClient,
) -> tuple[str, Callable, Optional[Callable], Optional[Callable], Optional[Callable]]:
    """Returns configuration tuple for the given API client.

    Args:
        client: An instance of OpenAI, Together, or Anthropic client

    Returns:
        tuple: (span_name, create_method, responses_method, stream_method, beta_parse_method)
            - span_name: String identifier for tracing
            - create_method: Reference to the client's creation method
            - responses_method: Reference to the client's responses method (if applicable)
            - stream_method: Reference to the client's stream method (if applicable)
            - beta_parse_method: Reference to the client's beta parse method (if applicable)

    Raises:
        ValueError: If client type is not supported
    """
    if isinstance(client, (OpenAI, AsyncOpenAI)):
        return (
            "OPENAI_API_CALL",
            client.chat.completions.create,
            client.responses.create,
            None,
            client.beta.chat.completions.parse,
        )
    elif isinstance(client, (Together, AsyncTogether)):
        return "TOGETHER_API_CALL", client.chat.completions.create, None, None, None
    elif isinstance(client, (Anthropic, AsyncAnthropic)):
        return (
            "ANTHROPIC_API_CALL",
            client.messages.create,
            None,
            client.messages.stream,
            None,
        )
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        return "GOOGLE_API_CALL", client.models.generate_content, None, None, None
    raise ValueError(f"Unsupported client type: {type(client)}")


def _format_input_data(client: ApiClient, **kwargs) -> dict:
    """Format input parameters based on client type.

    Extracts relevant parameters from kwargs based on the client type
    to ensure consistent tracing across different APIs.
    """
    if isinstance(client, (OpenAI, Together, AsyncOpenAI, AsyncTogether)):
        input_data = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
        }
        if kwargs.get("response_format"):
            input_data["response_format"] = kwargs.get("response_format")
        return input_data
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        return {"model": kwargs.get("model"), "contents": kwargs.get("contents")}
    # Anthropic requires additional max_tokens parameter
    return {
        "model": kwargs.get("model"),
        "messages": kwargs.get("messages"),
        "max_tokens": kwargs.get("max_tokens"),
    }


def _format_response_output_data(client: ApiClient, response: Any) -> tuple:
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
        return None, None

    prompt_cost, completion_cost = cost_per_token(
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    total_cost_usd = (
        (prompt_cost + completion_cost) if prompt_cost and completion_cost else None
    )
    usage = TraceUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_cost_usd=prompt_cost,
        completion_tokens_cost_usd=completion_cost,
        total_cost_usd=total_cost_usd,
        model_name=model_name,
    )
    return message_content, usage


def _format_output_data(
    client: ApiClient, response: Any
) -> tuple[Optional[str], Optional[TraceUsage]]:
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
        if (
            hasattr(response.choices[0].message, "parsed")
            and response.choices[0].message.parsed
        ):
            message_content = response.choices[0].message.parsed
        else:
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
    total_cost_usd = (
        (prompt_cost + completion_cost) if prompt_cost and completion_cost else None
    )
    usage = TraceUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_cost_usd=prompt_cost,
        completion_tokens_cost_usd=completion_cost,
        total_cost_usd=total_cost_usd,
        model_name=model_name,
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
    except Exception:
        # Fallback if signature inspection fails
        return {**{f"arg{i}": arg for i, arg in enumerate(args)}, **kwargs}


# NOTE: This builds once, can be tweaked if we are missing / capturing other unncessary modules
# @link https://docs.python.org/3.13/library/sysconfig.html
_TRACE_FILEPATH_BLOCKLIST = tuple(
    os.path.realpath(p) + os.sep
    for p in {
        sysconfig.get_paths()["stdlib"],
        sysconfig.get_paths().get("platstdlib", ""),
        *site.getsitepackages(),
        site.getusersitepackages(),
        *(
            [os.path.join(os.path.dirname(__file__), "../../judgeval/")]
            if os.environ.get("JUDGMENT_DEV")
            else []
        ),
    }
    if p
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
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                return chunk.candidates[0].content.parts[0].text
    except (AttributeError, IndexError, KeyError):
        # Handle cases where chunk structure is unexpected or doesn't contain content
        pass  # Return None
    return None


def _extract_usage_from_final_chunk(
    client: ApiClient, chunk: Any
) -> Optional[Dict[str, int]]:
    """Extracts usage data if present in the *final* chunk (client-specific)."""
    try:
        # OpenAI/Together include usage in the *last* chunk's `usage` attribute if available
        # This typically requires specific API versions or settings. Often usage is *not* streamed.
        if isinstance(client, (OpenAI, Together, AsyncOpenAI, AsyncTogether)):
            # Check if usage is directly on the chunk (some models might do this)
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            # Check if usage is nested within choices (less common for final chunk, but check)
            elif (
                chunk.choices
                and hasattr(chunk.choices[0], "usage")
                and chunk.choices[0].usage
            ):
                prompt_tokens = chunk.choices[0].usage.prompt_tokens
                completion_tokens = chunk.choices[0].usage.completion_tokens

            prompt_cost, completion_cost = cost_per_token(
                model=chunk.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            total_cost_usd = (
                (prompt_cost + completion_cost)
                if prompt_cost and completion_cost
                else None
            )
            return TraceUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
                prompt_tokens_cost_usd=prompt_cost,
                completion_tokens_cost_usd=completion_cost,
                total_cost_usd=total_cost_usd,
                model_name=chunk.model,
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
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                return {
                    "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                    "completion_tokens": chunk.usage_metadata.candidates_token_count,
                    "total_tokens": chunk.usage_metadata.total_token_count,
                }

    except (AttributeError, IndexError, KeyError, TypeError):
        # Handle cases where usage data is missing or malformed
        pass  # Return None
    return None


# --- Sync Stream Wrapper ---
def _sync_stream_wrapper(
    original_stream: Iterator,
    client: ApiClient,
    span: TraceSpan,
    trace_across_async_contexts: bool = Tracer.trace_across_async_contexts,
) -> Generator[Any, None, None]:
    """Wraps a synchronous stream iterator to capture content and update the trace."""
    content_parts = []  # Use a list instead of string concatenation
    final_usage = None
    last_chunk = None
    try:
        for chunk in original_stream:
            content_part = _extract_content_from_chunk(client, chunk)
            if content_part:
                content_parts.append(
                    content_part
                )  # Append to list instead of concatenating
            last_chunk = chunk  # Keep track of the last chunk for potential usage data
            yield chunk  # Pass the chunk to the caller
    finally:
        # Attempt to extract usage from the last chunk received
        if last_chunk:
            final_usage = _extract_usage_from_final_chunk(client, last_chunk)

        # Update the trace entry with the accumulated content and usage
        span.output = "".join(content_parts)
        span.usage = final_usage

        # Queue the completed LLM span now that streaming is done and all data is available

        current_trace = _get_current_trace(trace_across_async_contexts)
        if current_trace and current_trace.background_span_service:
            current_trace.background_span_service.queue_span(
                span, span_state="completed"
            )

        # Note: We might need to adjust _serialize_output if this dict causes issues,
        # but Pydantic's model_dump should handle dicts.


# --- Async Stream Wrapper ---
async def _async_stream_wrapper(
    original_stream: AsyncIterator,
    client: ApiClient,
    span: TraceSpan,
    trace_across_async_contexts: bool = Tracer.trace_across_async_contexts,
) -> AsyncGenerator[Any, None]:
    # [Existing logic - unchanged]
    content_parts = []  # Use a list instead of string concatenation
    final_usage_data = None
    last_content_chunk = None
    anthropic_input_tokens = 0
    anthropic_output_tokens = 0

    try:
        model_name = ""
        async for chunk in original_stream:
            # Check for OpenAI's final usage chunk
            if (
                isinstance(client, (AsyncOpenAI, OpenAI))
                and hasattr(chunk, "usage")
                and chunk.usage is not None
            ):
                final_usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
                model_name = chunk.model
                yield chunk
                continue

            if isinstance(client, (AsyncAnthropic, Anthropic)) and hasattr(
                chunk, "type"
            ):
                if chunk.type == "message_start":
                    if (
                        hasattr(chunk, "message")
                        and hasattr(chunk.message, "usage")
                        and hasattr(chunk.message.usage, "input_tokens")
                    ):
                        anthropic_input_tokens = chunk.message.usage.input_tokens
                        model_name = chunk.message.model
                elif chunk.type == "message_delta":
                    if hasattr(chunk, "usage") and hasattr(
                        chunk.usage, "output_tokens"
                    ):
                        anthropic_output_tokens = chunk.usage.output_tokens

            content_part = _extract_content_from_chunk(client, chunk)
            if content_part:
                content_parts.append(
                    content_part
                )  # Append to list instead of concatenating
                last_content_chunk = chunk

            yield chunk
    finally:
        anthropic_final_usage = None
        if isinstance(client, (AsyncAnthropic, Anthropic)) and (
            anthropic_input_tokens > 0 or anthropic_output_tokens > 0
        ):
            anthropic_final_usage = {
                "prompt_tokens": anthropic_input_tokens,
                "completion_tokens": anthropic_output_tokens,
                "total_tokens": anthropic_input_tokens + anthropic_output_tokens,
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
                model_name=model_name,
            )
        if span and hasattr(span, "output"):
            span.output = "".join(content_parts)
            span.usage = usage_info
            start_ts = getattr(span, "created_at", time.time())
            span.duration = time.time() - start_ts

            # Queue the completed LLM span now that async streaming is done and all data is available
            current_trace = _get_current_trace(trace_across_async_contexts)
            if current_trace and current_trace.background_span_service:
                current_trace.background_span_service.queue_span(
                    span, span_state="completed"
                )
        # else: # Handle error case if necessary, but remove debug print


def cost_per_token(*args, **kwargs):
    try:
        return _original_cost_per_token(*args, **kwargs)
    except Exception as e:
        warnings.warn(f"Error calculating cost per token: {e}")
        return None, None


class _BaseStreamManagerWrapper:
    def __init__(
        self,
        original_manager,
        client,
        span_name,
        trace_client,
        stream_wrapper_func,
        input_kwargs,
        trace_across_async_contexts: bool = Tracer.trace_across_async_contexts,
    ):
        self._original_manager = original_manager
        self._client = client
        self._span_name = span_name
        self._trace_client = trace_client
        self._stream_wrapper_func = stream_wrapper_func
        self._input_kwargs = input_kwargs
        self._parent_span_id_at_entry = None
        self._trace_across_async_contexts = trace_across_async_contexts

    def _create_span(self):
        start_time = time.time()
        span_id = str(uuid.uuid4())
        current_depth = 0
        if (
            self._parent_span_id_at_entry
            and self._parent_span_id_at_entry in self._trace_client._span_depths
        ):
            current_depth = (
                self._trace_client._span_depths[self._parent_span_id_at_entry] + 1
            )
        self._trace_client._span_depths[span_id] = current_depth
        span = TraceSpan(
            function=self._span_name,
            span_id=span_id,
            trace_id=self._trace_client.trace_id,
            depth=current_depth,
            message=self._span_name,
            created_at=start_time,
            span_type="llm",
            parent_span_id=self._parent_span_id_at_entry,
        )
        self._trace_client.add_span(span)
        return span_id, span

    def _finalize_span(self, span_id):
        span = self._trace_client.span_id_to_span.get(span_id)
        if span:
            span.duration = time.time() - span.created_at
        if span_id in self._trace_client._span_depths:
            del self._trace_client._span_depths[span_id]


class _TracedAsyncStreamManagerWrapper(
    _BaseStreamManagerWrapper, AbstractAsyncContextManager
):
    async def __aenter__(self):
        self._parent_span_id_at_entry = self._trace_client.get_current_span()
        if not self._trace_client:
            return await self._original_manager.__aenter__()

        span_id, span = self._create_span()
        self._span_context_token = self._trace_client.set_current_span(span_id)
        span.inputs = _format_input_data(self._client, **self._input_kwargs)

        # Call the original __aenter__ and expect it to be an async generator
        raw_iterator = await self._original_manager.__aenter__()
        span.output = "<pending stream>"
        return self._stream_wrapper_func(
            raw_iterator, self._client, span, self._trace_across_async_contexts
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_span_context_token"):
            span_id = self._trace_client.get_current_span()
            self._finalize_span(span_id)
            self._trace_client.reset_current_span(self._span_context_token)
            delattr(self, "_span_context_token")
        return await self._original_manager.__aexit__(exc_type, exc_val, exc_tb)


class _TracedSyncStreamManagerWrapper(
    _BaseStreamManagerWrapper, AbstractContextManager
):
    def __enter__(self):
        self._parent_span_id_at_entry = self._trace_client.get_current_span()
        if not self._trace_client:
            return self._original_manager.__enter__()

        span_id, span = self._create_span()
        self._span_context_token = self._trace_client.set_current_span(span_id)
        span.inputs = _format_input_data(self._client, **self._input_kwargs)

        raw_iterator = self._original_manager.__enter__()
        span.output = "<pending stream>"
        return self._stream_wrapper_func(
            raw_iterator, self._client, span, self._trace_across_async_contexts
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_span_context_token"):
            span_id = self._trace_client.get_current_span()
            self._finalize_span(span_id)
            self._trace_client.reset_current_span(self._span_context_token)
            delattr(self, "_span_context_token")
        return self._original_manager.__exit__(exc_type, exc_val, exc_tb)


# --- Helper function for instance-prefixed qual_name ---
def get_instance_prefixed_name(instance, class_name, class_identifiers):
    """
    Returns the agent name (prefix) if the class and attribute are found in class_identifiers.
    Otherwise, returns None.
    """
    if class_name in class_identifiers:
        class_config = class_identifiers[class_name]
        attr = class_config["identifier"]

        if hasattr(instance, attr):
            instance_name = getattr(instance, attr)
            return instance_name
        else:
            raise Exception(
                f"Attribute {attr} does not exist for {class_name}. Check your identify() decorator."
            )
    return None
