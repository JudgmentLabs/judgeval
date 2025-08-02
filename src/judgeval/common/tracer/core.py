"""
Tracing system for judgeval that allows for function tracing using decorators.
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import inspect
import os
import threading
import time
import traceback
import uuid
import contextvars
import sys
from contextlib import (
    contextmanager,
)
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    TypeAlias,
)
import types
import random


from judgeval.common.tracer.constants import _TRACE_FILEPATH_BLOCKLIST

from judgeval.common.tracer.otel_span_processor import JudgmentSpanProcessor
from judgeval.common.tracer.span_processor import SpanProcessorBase
from judgeval.common.tracer.trace_manager import TraceManagerClient
from litellm import cost_per_token as _original_cost_per_token
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.responses.response import Response
from openai.types.chat import ParsedChatCompletion
from together import Together, AsyncTogether
from anthropic import Anthropic, AsyncAnthropic
from google import genai
from groq import Groq, AsyncGroq

from judgeval.data import Example, Trace, TraceSpan, TraceUsage
from judgeval.scorers import APIScorerConfig, BaseScorer
from judgeval.evaluation_run import EvaluationRun
from judgeval.local_eval_queue import LocalEvaluationQueue
from judgeval.common.api import JudgmentApiClient
from judgeval.common.utils import ExcInfo, validate_api_key
from judgeval.common.logger import judgeval_logger
from judgeval.constants import DEFAULT_GPT_MODEL


current_trace_var = contextvars.ContextVar[Optional["TraceClient"]](
    "current_trace", default=None
)
current_span_var = contextvars.ContextVar[Optional[str]]("current_span", default=None)

ApiClient: TypeAlias = Union[
    OpenAI,
    Together,
    Anthropic,
    AsyncOpenAI,
    AsyncAnthropic,
    AsyncTogether,
    genai.Client,
    genai.client.AsyncClient,
    Groq,
    AsyncGroq,
]
SpanType: TypeAlias = str


class TraceClient:
    """Client for managing a single trace context"""

    def __init__(
        self,
        tracer: Tracer,
        trace_id: Optional[str] = None,
        name: str = "default",
        project_name: Union[str, None] = None,
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
        self.customer_id: Optional[str] = None
        self.tags: List[Union[str, set, tuple]] = []
        self.metadata: Dict[str, Any] = {}
        self.has_notification: Optional[bool] = False
        self.update_id: int = 1
        self.trace_spans: List[TraceSpan] = []
        self.span_id_to_span: Dict[str, TraceSpan] = {}
        self.evaluation_runs: List[EvaluationRun] = []
        self.start_time: Optional[float] = None
        self.trace_manager_client = TraceManagerClient(
            tracer.api_key, tracer.organization_id, tracer
        )
        self._span_depths: Dict[str, int] = {}

        self.otel_span_processor = tracer.otel_span_processor

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
                self.save(final_save=False)
            except Exception as e:
                judgeval_logger.warning(
                    f"Failed to save initial trace for live tracking: {e}"
                )
        start_time = time.time()

        span_id = str(uuid.uuid4())

        parent_span_id = self.get_current_span()
        token = self.set_current_span(span_id)

        current_depth = 0
        if parent_span_id and parent_span_id in self._span_depths:
            current_depth = self._span_depths[parent_span_id] + 1

        self._span_depths[span_id] = current_depth

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

        self.otel_span_processor.queue_span_update(span, span_state="input")

        try:
            yield self
        finally:
            duration = time.time() - start_time
            span.duration = duration

            self.otel_span_processor.queue_span_update(span, span_state="completed")

            if span_id in self._span_depths:
                del self._span_depths[span_id]
            self.reset_current_span(token)

    def async_evaluate(
        self,
        scorer: Union[APIScorerConfig, BaseScorer],
        example: Example,
        model: str = DEFAULT_GPT_MODEL,
    ):
        start_time = time.time()
        span_id = self.get_current_span()
        eval_run_name = (
            f"{self.name.capitalize()}-{span_id}-{scorer.score_type.capitalize()}"
        )

        if isinstance(scorer, APIScorerConfig):
            eval_run = EvaluationRun(
                organization_id=self.tracer.organization_id,
                project_name=self.project_name,
                eval_name=eval_run_name,
                examples=[example],
                scorers=[scorer],
                model=model,
                judgment_api_key=self.tracer.api_key,
                trace_span_id=span_id,
            )

            self.add_eval_run(eval_run, start_time)

            if span_id:
                current_span = self.span_id_to_span.get(span_id)
                if current_span:
                    self.otel_span_processor.queue_evaluation_run(
                        eval_run, span_id=span_id, span_data=current_span
                    )
        elif isinstance(scorer, BaseScorer):
            # Handle custom scorers using local evaluation queue
            eval_run = EvaluationRun(
                organization_id=self.tracer.organization_id,
                project_name=self.project_name,
                eval_name=eval_run_name,
                examples=[example],
                scorers=[scorer],
                model=model,
                judgment_api_key=self.tracer.api_key,
                trace_span_id=span_id,
                trace_id=self.trace_id,
            )

            self.add_eval_run(eval_run, start_time)

            # Enqueue the evaluation run to the local evaluation queue
            self.tracer.local_eval_queue.enqueue(eval_run)

    def add_eval_run(self, eval_run: EvaluationRun, start_time: float):
        current_span_id = eval_run.trace_span_id

        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.has_evaluation = True
        self.evaluation_runs.append(eval_run)

    def record_input(self, inputs: dict):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            if "self" in inputs:
                del inputs["self"]
            span.inputs = inputs

            try:
                self.otel_span_processor.queue_span_update(span, span_state="input")
            except Exception as e:
                judgeval_logger.warning(f"Failed to queue span with input data: {e}")

    def record_agent_name(self, agent_name: str):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.agent_name = agent_name

            self.otel_span_processor.queue_span_update(span, span_state="agent_name")

    def record_state_before(self, state: dict):
        """Records the agent's state before a tool execution on the current span.

        Args:
            state: A dictionary representing the agent's state.
        """
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.state_before = state

            self.otel_span_processor.queue_span_update(span, span_state="state_before")

    def record_state_after(self, state: dict):
        """Records the agent's state after a tool execution on the current span.

        Args:
            state: A dictionary representing the agent's state.
        """
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.state_after = state

            self.otel_span_processor.queue_span_update(span, span_state="state_after")

    async def _update_coroutine(self, span: TraceSpan, coroutine: Any, field: str):
        """Helper method to update the output of a trace entry once the coroutine completes"""
        try:
            result = await coroutine
            setattr(span, field, result)

            if field == "output":
                self.otel_span_processor.queue_span_update(span, span_state="output")

            return result
        except Exception as e:
            setattr(span, field, f"Error: {str(e)}")

            if field == "output":
                self.otel_span_processor.queue_span_update(span, span_state="output")

            raise

    def record_output(self, output: Any):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.output = "<pending>" if inspect.iscoroutine(output) else output

            if inspect.iscoroutine(output):
                asyncio.create_task(self._update_coroutine(span, output, "output"))

            if not inspect.iscoroutine(output):
                self.otel_span_processor.queue_span_update(span, span_state="output")

            return span
        return None

    def record_usage(self, usage: TraceUsage):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.usage = usage

            self.otel_span_processor.queue_span_update(span, span_state="usage")

            return span
        return None

    def record_error(self, error: Dict[str, Any]):
        current_span_id = self.get_current_span()
        if current_span_id:
            span = self.span_id_to_span[current_span_id]
            span.error = error

            self.otel_span_processor.queue_span_update(span, span_state="error")

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
            return 0.0
        return time.time() - self.start_time

    def save(self, final_save: bool = False) -> Tuple[str, dict]:
        """
        Save the current trace to the database with rate limiting checks.
        First checks usage limits, then upserts the trace if allowed.

        Args:
            final_save: Whether this is the final save (updates usage counters)

        Returns a tuple of (trace_id, server_response) where server_response contains the UI URL and other metadata.
        """
        if final_save:
            try:
                self.otel_span_processor.flush_pending_spans()
            except Exception as e:
                judgeval_logger.warning(
                    f"Error flushing spans for trace {self.trace_id}: {e}"
                )

        total_duration = self.get_duration()

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

        server_response = self.trace_manager_client.upsert_trace(
            trace_data,
            offline_mode=self.tracer.offline_mode,
            show_link=not final_save,
            final_save=final_save,
        )

        if self.start_time is None:
            self.start_time = time.time()

        self.update_id += 1

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

    def set_reward_score(self, reward_score: Union[float, Dict[str, float]]):
        """
        Set the reward score for this trace to be used for RL or SFT.

        Args:
            reward_score: The reward score to set
        """
        self.update_metadata({"reward_score": reward_score})


def _capture_exception_for_trace(
    current_trace: Optional[TraceClient], exc_info: ExcInfo
):
    if not current_trace:
        return

    exc_type, exc_value, exc_traceback_obj = exc_info
    formatted_exception = {
        "type": exc_type.__name__ if exc_type else "UnknownExceptionType",
        "message": str(exc_value) if exc_value else "No exception message",
        "traceback": (
            traceback.format_tb(exc_traceback_obj) if exc_traceback_obj else []
        ),
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
                pass

        our_result = self._trace(frame, event, arg, self._cooperative_sys_trace)

        if original_result is None and self._original_sys_trace:
            return None

        return our_result or original_result

    def _cooperative_threading_trace(
        self, frame: types.FrameType, event: str, arg: Any
    ):
        """Cooperative trace function for threading.settrace that chains with existing tracers."""
        original_result = None
        if self._original_threading_trace:
            try:
                original_result = self._original_threading_trace(frame, event, arg)
            except Exception:
                pass

        our_result = self._trace(frame, event, arg, self._cooperative_threading_trace)

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
        api_key: Union[str, None] = os.getenv("JUDGMENT_API_KEY"),
        organization_id: Union[str, None] = os.getenv("JUDGMENT_ORG_ID"),
        project_name: Union[str, None] = None,
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
        span_batch_size: int = 50,
        span_flush_interval: float = 1.0,
        span_max_queue_size: int = 2048,
        span_export_timeout: int = 30000,
    ):
        try:
            if not api_key:
                raise ValueError(
                    "api_key parameter must be provided. Please provide a valid API key value or set the JUDGMENT_API_KEY environment variable"
                )

            if not organization_id:
                raise ValueError(
                    "organization_id parameter must be provided. Please provide a valid organization ID value or set the JUDGMENT_ORG_ID environment variable"
                )

            try:
                result, response = validate_api_key(api_key)
            except Exception as e:
                judgeval_logger.error(
                    f"Issue with verifying API key, disabling monitoring: {e}"
                )
                enable_monitoring = False
                result = True

            if not result:
                raise ValueError(f"Issue with passed in Judgment API key: {response}")

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
            self.span_id_to_previous_span_id: Dict[str, Union[str, None]] = {}
            self.trace_id_to_previous_trace: Dict[str, Union[TraceClient, None]] = {}
            self.current_span_id: Optional[str] = None
            self.current_trace: Optional[TraceClient] = None
            self.trace_across_async_contexts: bool = trace_across_async_contexts
            Tracer.trace_across_async_contexts = trace_across_async_contexts

            # Initialize S3 storage if enabled
            self.use_s3 = use_s3
            if use_s3:
                from judgeval.common.storage.s3_storage import S3Storage

                try:
                    self.s3_storage = S3Storage(
                        bucket_name=s3_bucket_name,
                        aws_access_key_id=s3_aws_access_key_id,
                        aws_secret_access_key=s3_aws_secret_access_key,
                        region_name=s3_region_name,
                    )
                except Exception as e:
                    judgeval_logger.error(
                        f"Issue with initializing S3 storage, disabling S3: {e}"
                    )
                    self.use_s3 = False

            self.offline_mode = False  # This is used to differentiate traces between online and offline (IE experiments vs monitoring page)
            self.deep_tracing: bool = deep_tracing

            self.span_batch_size = span_batch_size
            self.span_flush_interval = span_flush_interval
            self.span_max_queue_size = span_max_queue_size
            self.span_export_timeout = span_export_timeout
            self.otel_span_processor: SpanProcessorBase
            if enable_monitoring:
                self.otel_span_processor = JudgmentSpanProcessor(
                    judgment_api_key=api_key,
                    organization_id=organization_id,
                    batch_size=span_batch_size,
                    flush_interval=span_flush_interval,
                    max_queue_size=span_max_queue_size,
                    export_timeout=span_export_timeout,
                )
            else:
                self.otel_span_processor = SpanProcessorBase()

            # Initialize local evaluation queue for custom scorers
            self.local_eval_queue = LocalEvaluationQueue()

            # Start workers with callback to log results only if monitoring is enabled
            if enable_evaluations and enable_monitoring:
                self.local_eval_queue.start_workers(
                    callback=self._log_eval_results_callback
                )

            atexit.register(self._cleanup_on_exit)
        except Exception as e:
            judgeval_logger.error(
                f"Issue with initializing Tracer: {e}. Disabling monitoring and evaluations."
            )
            self.enable_monitoring = False
            self.enable_evaluations = False

    def set_current_span(
        self, span_id: str
    ) -> Optional[contextvars.Token[Union[str, None]]]:
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
        token: Optional[contextvars.Token[Union[str, None]]] = None,
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
    ) -> Optional[contextvars.Token[Union[TraceClient, None]]]:
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
        token: Optional[contextvars.Token[Union[TraceClient, None]]] = None,
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
        self, name: str, project_name: Union[str, None] = None
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

        with trace.span(name or "unnamed_trace"):
            try:
                # Save the trace to the database to handle Evaluations' trace_id referential integrity
                yield trace
            finally:
                # Reset the context variable
                self.reset_current_trace(token)

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

                current_trace = self.get_current_trace()

                if not current_trace:
                    trace_id = str(uuid.uuid4())
                    project = self.project_name

                    current_trace = TraceClient(
                        self,
                        trace_id,
                        span_name,
                        project_name=project,
                        enable_monitoring=self.enable_monitoring,
                        enable_evaluations=self.enable_evaluations,
                    )

                    trace_token = self.set_current_trace(current_trace)

                    try:
                        with current_trace.span(span_name, span_type=span_type) as span:
                            inputs = combine_args_kwargs(func, args, kwargs)
                            span.record_input(inputs)
                            if agent_name:
                                span.record_agent_name(agent_name)

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

                            self._conditionally_capture_and_record_state(
                                span, args, is_before=False
                            )

                            span.record_output(result)
                        return result
                    finally:
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

                            trace_id, server_response = current_trace.save(
                                final_save=True
                            )

                            self.traces.append(complete_trace_data)

                            self.reset_current_trace(trace_token)
                        except Exception as e:
                            judgeval_logger.warning(f"Issue with async_wrapper: {e}")
                            pass
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
                        span_name,
                        project_name=project,
                        enable_monitoring=self.enable_monitoring,
                        enable_evaluations=self.enable_evaluations,
                    )

                    trace_token = self.set_current_trace(current_trace)

                    try:
                        with current_trace.span(span_name, span_type=span_type) as span:
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
                        try:
                            trace_id, server_response = current_trace.save(
                                final_save=True
                            )

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
                            self.reset_current_trace(trace_token)
                        except Exception as e:
                            judgeval_logger.warning(f"Issue with save: {e}")
                            pass
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
                        judgeval_logger.info(
                            f"{cls.__name__}.{name} already decorated, skipping"
                        )
                    continue

                try:
                    decorated_method = self.observe(method, span_type="tool")
                    setattr(cls, name, decorated_method)
                    decorated.append(name)
                except Exception as e:
                    if warn_on_double_decoration:
                        judgeval_logger.warning(
                            f"Failed to decorate {cls.__name__}.{name}: {e}"
                        )

            return cls

        return decorate_class if cls is None else decorate_class(cls)

    def async_evaluate(
        self,
        scorer: Union[APIScorerConfig, BaseScorer],
        example: Example,
        model: str = DEFAULT_GPT_MODEL,
        sampling_rate: float = 1,
        background: bool = True,
    ):
        try:
            if not self.enable_monitoring or not self.enable_evaluations:
                return

            if not isinstance(scorer, (APIScorerConfig, BaseScorer)):
                judgeval_logger.warning(
                    f"Scorer must be an instance of APIScorerConfig or BaseScorer, got {type(scorer)}, skipping evaluation"
                )
                return

            if not isinstance(example, Example):
                judgeval_logger.warning(
                    f"Example must be an instance of Example, got {type(example)} skipping evaluation"
                )
                return

            if sampling_rate < 0:
                judgeval_logger.warning(
                    "Cannot set sampling_rate below 0, skipping evaluation"
                )
                return

            if sampling_rate > 1:
                judgeval_logger.warning(
                    "Cannot set sampling_rate above 1, skipping evaluation"
                )
                return

            percentage = random.uniform(0, 1)
            if percentage > sampling_rate:
                judgeval_logger.info("Skipping async_evaluate due to sampling rate")
                return

            current_trace = self.get_current_trace()
            if current_trace:
                current_trace.async_evaluate(
                    scorer=scorer, example=example, model=model
                )
            else:
                judgeval_logger.warning(
                    "No trace found (context var or fallback), skipping evaluation"
                )
        except Exception as e:
            judgeval_logger.warning(f"Issue with async_evaluate: {e}")

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
            judgeval_logger.warning("No current trace found, cannot set metadata")

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
            judgeval_logger.warning("No current trace found, cannot set customer ID")

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
            judgeval_logger.warning("No current trace found, cannot set tags")

    def set_reward_score(self, reward_score: Union[float, Dict[str, float]]):
        """
        Set the reward score for this trace to be used for RL or SFT.

        Args:
            reward_score: The reward score to set
        """
        current_trace = self.get_current_trace()
        if current_trace:
            current_trace.set_reward_score(reward_score)
        else:
            judgeval_logger.warning("No current trace found, cannot set reward score")

    def get_otel_span_processor(self) -> SpanProcessorBase:
        """Get the OpenTelemetry span processor instance."""
        return self.otel_span_processor

    def flush_background_spans(self, timeout_millis: int = 30000):
        """Flush all pending spans in the background service."""
        self.otel_span_processor.force_flush(timeout_millis)

    def shutdown_background_service(self):
        """Shutdown the background span service."""
        self.otel_span_processor.shutdown()
        self.otel_span_processor = SpanProcessorBase()

    def wait_for_completion(self, timeout: Optional[float] = 30.0) -> bool:
        """Wait for all evaluations and span processing to complete.

        This method blocks until all queued evaluations are processed and
        all pending spans are flushed to the server.

        Args:
            timeout: Maximum time to wait in seconds. Defaults to 30 seconds.
                    None means wait indefinitely.

        Returns:
            True if all processing completed within the timeout, False otherwise.

        """
        try:
            judgeval_logger.info("Waiting for all evaluations and spans to complete...")

            # Wait for all queued evaluation work to complete
            eval_completed = self.local_eval_queue.wait_for_completion()
            if not eval_completed:
                judgeval_logger.warning(
                    f"Local evaluation queue did not complete within {timeout} seconds"
                )
                return False

            self.flush_background_spans()

            judgeval_logger.info("All evaluations and spans completed successfully")
            return True

        except Exception as e:
            judgeval_logger.warning(f"Error while waiting for completion: {e}")
            return False

    def _log_eval_results_callback(self, evaluation_run, scoring_results):
        """Callback to log evaluation results after local processing."""
        try:
            if scoring_results and self.enable_evaluations and self.enable_monitoring:
                # Convert scoring results to the format expected by API client
                results_dict = [
                    result.model_dump(warnings=False) for result in scoring_results
                ]
                api_client = JudgmentApiClient(self.api_key, self.organization_id)
                api_client.log_evaluation_results(
                    results_dict, evaluation_run.model_dump(warnings=False)
                )
        except Exception as e:
            judgeval_logger.warning(f"Failed to log local evaluation results: {e}")

    def _cleanup_on_exit(self):
        """Cleanup handler called on application exit to ensure spans are flushed."""
        judgeval_logger.info("Tracer cleanup on exit started")
        try:
            # Wait for all queued evaluation work to complete before stopping
            judgeval_logger.info("Waiting for local evaluation queue to complete...")
            completed = self.local_eval_queue.wait_for_completion()
            if not completed:
                judgeval_logger.warning(
                    "Local evaluation queue did not complete within 30 seconds"
                )

            self.local_eval_queue.stop_workers()
            self.flush_background_spans()
        except Exception as e:
            judgeval_logger.warning(f"Error during tracer cleanup: {e}")
        finally:
            try:
                self.shutdown_background_service()
            except Exception as e:
                judgeval_logger.warning(
                    f"Error during background service shutdown: {e}"
                )
        judgeval_logger.info("Tracer cleanup on exit completed")


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

    def process_span(span, response):
        """Format and record the output in the span"""
        output, usage = _format_output_data(client, response)
        span.record_output(output)
        span.record_usage(usage)

        return response

    def wrapped(function):
        def wrapper(*args, **kwargs):
            current_trace = _get_current_trace(trace_across_async_contexts)
            if not current_trace:
                return function(*args, **kwargs)

            with current_trace.span(span_name, span_type="llm") as span:
                span.record_input(kwargs)

                try:
                    response = function(*args, **kwargs)
                    return process_span(span, response)
                except Exception as e:
                    _capture_exception_for_trace(span, sys.exc_info())
                    raise e

        return wrapper

    def wrapped_async(function):
        async def wrapper(*args, **kwargs):
            current_trace = _get_current_trace(trace_across_async_contexts)
            if not current_trace:
                return await function(*args, **kwargs)

            with current_trace.span(span_name, span_type="llm") as span:
                span.record_input(kwargs)

                try:
                    response = await function(*args, **kwargs)
                    return process_span(span, response)
                except Exception as e:
                    _capture_exception_for_trace(span, sys.exc_info())
                    raise e

        return wrapper

    if isinstance(client, (OpenAI)):
        setattr(client.chat.completions, "create", wrapped(original_create))
        setattr(client.responses, "create", wrapped(original_responses_create))
        setattr(client.beta.chat.completions, "parse", wrapped(original_beta_parse))
    elif isinstance(client, (AsyncOpenAI)):
        setattr(client.chat.completions, "create", wrapped_async(original_create))
        setattr(client.responses, "create", wrapped_async(original_responses_create))
        setattr(
            client.beta.chat.completions, "parse", wrapped_async(original_beta_parse)
        )
    elif isinstance(client, (Together)):
        setattr(client.chat.completions, "create", wrapped(original_create))
    elif isinstance(client, (AsyncTogether)):
        setattr(client.chat.completions, "create", wrapped_async(original_create))
    elif isinstance(client, (Anthropic)):
        setattr(client.messages, "create", wrapped(original_create))
    elif isinstance(client, (AsyncAnthropic)):
        setattr(client.messages, "create", wrapped_async(original_create))
    elif isinstance(client, (genai.Client)):
        setattr(client.models, "generate_content", wrapped(original_create))
    elif isinstance(client, (genai.client.AsyncClient)):
        setattr(client.models, "generate_content", wrapped_async(original_create))
    elif isinstance(client, (Groq)):
        setattr(client.chat.completions, "create", wrapped(original_create))
    elif isinstance(client, (AsyncGroq)):
        setattr(client.chat.completions, "create", wrapped_async(original_create))

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
    elif isinstance(client, (Groq, AsyncGroq)):
        return "GROQ_API_CALL", client.chat.completions.create, None, None, None
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
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    model_name = None
    message_content = None

    if isinstance(client, (OpenAI, AsyncOpenAI)):
        if isinstance(response, ChatCompletion):
            model_name = response.model
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = (
                response.usage.completion_tokens if response.usage else 0
            )
            cache_read_input_tokens = (
                response.usage.prompt_tokens_details.cached_tokens
                if response.usage
                and response.usage.prompt_tokens_details
                and response.usage.prompt_tokens_details.cached_tokens
                else 0
            )

            if isinstance(response, ParsedChatCompletion):
                message_content = response.choices[0].message.parsed
            else:
                message_content = response.choices[0].message.content
        elif isinstance(response, Response):
            model_name = response.model
            prompt_tokens = response.usage.input_tokens if response.usage else 0
            completion_tokens = response.usage.output_tokens if response.usage else 0
            cache_read_input_tokens = (
                response.usage.input_tokens_details.cached_tokens
                if response.usage and response.usage.input_tokens_details
                else 0
            )
            if hasattr(response.output[0], "content"):
                message_content = "".join(
                    seg.text
                    for seg in response.output[0].content
                    if hasattr(seg, "text")
                )

        # Note: LiteLLM seems to use cache_read_input_tokens to calculate the cost for OpenAI
    elif isinstance(client, (Together, AsyncTogether)):
        model_name = "together_ai/" + response.model
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        message_content = response.choices[0].message.content

        # As of 2025-07-14, Together does not do any input cache token tracking
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        model_name = response.model_version
        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        message_content = response.candidates[0].content.parts[0].text

        if hasattr(response.usage_metadata, "cached_content_token_count"):
            cache_read_input_tokens = response.usage_metadata.cached_content_token_count
    elif isinstance(client, (Anthropic, AsyncAnthropic)):
        model_name = response.model
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        cache_read_input_tokens = response.usage.cache_read_input_tokens
        cache_creation_input_tokens = response.usage.cache_creation_input_tokens
        message_content = response.content[0].text
    elif isinstance(client, (Groq, AsyncGroq)):
        model_name = "groq/" + response.model
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        message_content = response.choices[0].message.content
    else:
        judgeval_logger.warning(f"Unsupported client type: {type(client)}")
        return None, None

    prompt_cost, completion_cost = cost_per_token(
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )
    total_cost_usd = (
        (prompt_cost + completion_cost) if prompt_cost and completion_cost else None
    )
    usage = TraceUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
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


def cost_per_token(*args, **kwargs):
    try:
        prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = (
            _original_cost_per_token(*args, **kwargs)
        )
        if (
            prompt_tokens_cost_usd_dollar == 0
            and completion_tokens_cost_usd_dollar == 0
        ):
            judgeval_logger.warning("LiteLLM returned a total of 0 for cost per token")
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    except Exception as e:
        judgeval_logger.warning(f"Error calculating cost per token: {e}")
        return None, None


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
