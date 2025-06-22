from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
import functools
import logging
import os
import sys
import time
import traceback
from types import CodeType, FrameType
from typing import Any, Callable, Dict, List, Optional, Union, cast

from judgeval.common.storage.judgment_batched_storage import JudgmentBatchedStorage
from judgeval.common.storage.judgment_storage import JudgmentStorage
from judgeval.common.storage.storage import ABCStorage
from judgeval.common.tracer.constants import _TRACE_FILEPATH_BLOCKLIST
from judgeval.common.tracer.model import TraceSave, TraceSpan
from judgeval.common.tracer.types import (
    CurrentSpanEntry,
    CurrentSpanExit,
    CurrentSpanType,
    ExcInfo,
    OptExcInfo,
    TSpanResetTokens,
)
from judgeval.common.tracer.utils import (
    extract_inputs_from_entry_frame,
    get_span_exit_depth_map,
    new_id,
)


class TraceClient:
    """Client for managing trace contexts."""

    __slots__ = (
        "trace_id",
        "name",
        "project_id",
        "overwrite",
        "enable_monitoring",
        "enable_evaluations",
        "_current_span_var",
        "_span_reset_tokens_var",
        "_surpress_deep_tracing",
        "_code_should_trace",
        "_trace_spans",
        "_storage_client",
    )

    trace_id: str
    name: str
    project_id: Optional[str]
    overwrite: bool
    enable_monitoring: bool
    enable_evaluations: bool

    _current_span_var: ContextVar[Union[CurrentSpanType, None]]
    _span_reset_tokens_var: ContextVar[Union[TSpanResetTokens, None]]
    _surpress_deep_tracing: ContextVar[bool]

    _code_should_trace: set[int]
    _trace_spans: List[CurrentSpanExit]

    _storage_client: ABCStorage

    def __init__(
        self,
        trace_id: Optional[str] = None,
        name: str = "default",
        project_id: Optional[str] = None,
        overwrite: bool = False,
        enable_monitoring: bool = True,
        enable_evaluations: bool = True,
        storage_client: Optional[ABCStorage] = None,
    ):
        self.trace_id = trace_id if trace_id else new_id()
        self.name = name
        self.project_id = project_id
        self.overwrite = overwrite

        self.enable_monitoring = enable_monitoring
        self.enable_evaluations = enable_evaluations

        self._current_span_var = ContextVar[Union[CurrentSpanType, None]](
            "current_span", default=None
        )
        self._span_reset_tokens_var = ContextVar[Union[TSpanResetTokens, None]](
            "span_reset_tokens", default=None
        )
        self._surpress_deep_tracing = ContextVar[bool](
            "surpress_deep_tracing", default=False
        )

        self._code_should_trace = set()
        self._trace_spans = []

        self._storage_client = storage_client or JudgmentBatchedStorage()

    @property
    def current_span(self) -> Union[CurrentSpanType, None]:
        """Get the current span from the context variable."""
        return self._current_span_var.get()

    @current_span.setter
    def current_span(self, value: CurrentSpanType):
        self._current_span_var.set(value)

    @property
    def span_reset_tokens(self) -> TSpanResetTokens:
        """Get the reset tokens from the context variable."""
        span_reset_tokens = self._span_reset_tokens_var.get()
        if span_reset_tokens is None:
            self._span_reset_tokens_var.set({})
        return cast(TSpanResetTokens, self._span_reset_tokens_var.get())

    @span_reset_tokens.setter
    def span_reset_tokens(self, value: TSpanResetTokens):
        """Set the reset tokens in the context variable."""
        self._span_reset_tokens_var.set(value)

    def _do_trace_code(self, code: CodeType) -> None:
        """
        Register the code object for tracing.
        This is used to determine if the code should be traced or not.
        """
        self._code_should_trace.add(id(code))

    def _commit_current_span(self) -> None:
        """Commit the current span to the trace spans list."""
        if not isinstance(current_span := self.current_span, CurrentSpanExit):
            logging.warning(
                "Commit called without a completed current span entry. This should not happen."
            )
            return

        span_id = current_span.span_id
        self._trace_spans.append(current_span)

        if self.span_reset_tokens is None or span_id not in self.span_reset_tokens:
            logging.warning(
                f"Exit observed without a corresponding entry span for span_id: {span_id}. This should not happen."
            )
            return

        token = cast(Token[Any], self.span_reset_tokens[span_id])
        del self.span_reset_tokens[span_id]
        self._current_span_var.reset(token)

        logging.info(
            f"Span committed: {current_span.name} (id: {span_id}, parent_id: {current_span.parent_span_id}, "
            f"start_time: {current_span.start_time}, end_time: {current_span.end_time}, output: {current_span.output})"
        )

    def _get_formatted_exception_from_exc_info(self, exc_info: ExcInfo):
        """
        Format the exception information from the given exc_info tuple.
        This will extract the exception type, message, and traceback.
        For additional types of errors it will add specific properties
        like HTTP status codes and URLs if available.
        """
        exc_type, exc_value, exc_traceback_obj = exc_info
        formatted_exception: Dict[str, Any] = {
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
            status_code = getattr(
                getattr(exc_value, "response", None), "status_code", None
            )
            if status_code:
                formatted_exception["http"] = {
                    "url": url if url else "Unknown URL",
                    "status_code": status_code if status_code else None,
                }
        except Exception as e:
            pass

        return formatted_exception

    def observe_enter(self, name: str, inputs: Union[dict[str, Any], None]):
        span_id = new_id()
        parent_span_id = (
            self.current_span.span_id
            if isinstance(self.current_span, CurrentSpanEntry)
            else None
        )
        start_time = time.time()
        token = self._current_span_var.set(
            CurrentSpanEntry(
                name=name,
                span_id=span_id,
                span_type="function",
                parent_span_id=parent_span_id,
                start_time=start_time,
                inputs=inputs,
            )
        )
        if self.span_reset_tokens is None:
            self.span_reset_tokens = {}

        self.span_reset_tokens[span_id] = cast(Token[CurrentSpanEntry], token)
        logging.info(
            f"Span entered: {name} (id: {span_id}, parent_id: {parent_span_id}, start_time: {start_time}, inputs: {inputs})"
        )

    def observe_exit(self, output: Any):
        if not isinstance(current_span := self.current_span, CurrentSpanEntry):
            logging.warning(
                "Exit observed without a corresponding entry span. This should not happen."
            )
            return

        end_time = time.time()

        span_exit = CurrentSpanExit(
            name=current_span.name,
            span_id=current_span.span_id,
            span_type=current_span.span_type,
            parent_span_id=current_span.parent_span_id,
            start_time=current_span.start_time,
            end_time=end_time,
            inputs=current_span.inputs,
            output=output,
            error=current_span.error,
        )
        self.current_span = span_exit

        logging.info(
            f"Span exited: {span_exit.name} (id: {span_exit.span_id}, parent_id: {span_exit.parent_span_id}, "
            f"start_time: {span_exit.start_time}, end_time: {end_time}, output: {output})"
        )
        self._commit_current_span()

    def observe_error(self, exc_info: OptExcInfo):
        if not isinstance(current_span := self.current_span, CurrentSpanEntry):
            logging.warning(
                "Error observed without a corresponding entry span. This should not happen.",
            )

            return

        if not exc_info or exc_info == (None, None, None):
            logging.warning(
                "Error observed with no exception information. This should not happen."
            )
            return

        formatted_exc_info = self._get_formatted_exception_from_exc_info(
            cast(ExcInfo, exc_info)
        )

        self._current_span_var.set(
            CurrentSpanEntry(
                name=current_span.name,
                span_id=current_span.span_id,
                span_type=current_span.span_type,
                parent_span_id=current_span.parent_span_id,
                start_time=current_span.start_time,
                end_time=current_span.end_time,
                inputs=current_span.inputs,
                output=current_span.output,
                error=formatted_exc_info,
            )
        )

        logging.error(
            f"Error observed in span: {current_span.name} (id: {current_span.span_id}, "
            f"parent_id: {current_span.parent_span_id}, start_time: {current_span.start_time}, "
            f"error: {exc_info})"
        )

    @functools.cache
    def _is_user_code(self, filename: str):
        """
        Determines if the given filename corresponds to user code.
        """
        return (
            bool(filename)
            and not filename.startswith("<")
            and not os.path.realpath(filename).startswith(_TRACE_FILEPATH_BLOCKLIST)
        )

    def _trace_daemon_factory_(self, deep_tracing: bool = False):
        """
        Factory function to create a trace daemon.
        Based on the global Judgment config, this function will create an appropriate trace daemon to
        be setup with syshooks.
        """

        def _should_trace(frame: FrameType) -> bool:
            # If we are not deep tracing, we only trace if the code object is in the set of code_should_trace
            if not deep_tracing and id(frame.f_code) in self._code_should_trace:
                logging.debug(
                    f"Tracing {frame.f_code.co_name} at {frame.f_code.co_filename} "
                    f"(id: {id(frame.f_code)}) because it is in the code_should_trace set."
                )
                return True

            if not self._is_user_code(frame.f_code.co_filename):
                logging.debug(
                    f"Skipping trace for {frame.f_code.co_name} at {frame.f_code.co_filename} "
                    f"because it is not user code."
                )
                return False

            if deep_tracing:
                logging.debug(
                    f"Tracing {frame.f_code.co_name} at {frame.f_code.co_filename} "
                    f"(id: {id(frame.f_code)}) because deep tracing is enabled."
                )
                return True

            logging.debug(
                f"Skipping trace for {frame.f_code.co_name} at {frame.f_code.co_filename} "
                f"because it is not in the code_should_trace set."
                f" (id: {id(frame.f_code)})"
            )

            return False

        def __trace_daemon__(
            frame: FrameType, event: str, arg: Any
        ) -> Union[Callable, None]:
            """
            The trace daemon that will be used to handle tracing.
            This function should be set up with sys.settrace / sys.setprofile hooks
            """
            if event not in ("call", "return", "exception"):
                return None

            if not _should_trace(frame):
                return None

            frame.f_trace_lines = False
            frame.f_trace_opcodes = False

            logging.info(
                f"Tracing {event} for {frame.f_code.co_name} at {frame.f_code.co_filename} "
                f"(id: {id(frame.f_code)})"
            )

            if event == "call":
                name = frame.f_code.co_qualname
                inputs = extract_inputs_from_entry_frame(frame)
                self.observe_enter(name, inputs)
                return __trace_daemon__

            elif event == "return":
                self.observe_exit(arg)

            elif event == "exception":
                self.observe_error(arg)

            return None

        return __trace_daemon__

    @contextmanager
    def span(self, name: str, span_type: str = "span"):
        """
        Context manager to create a span with the given name and type.
        This will automatically handle entering and exiting the span.
        """
        try:
            self.observe_enter(name, None)
            yield
        except Exception as e:
            self.observe_error(sys.exc_info())
            raise e
        finally:
            self.observe_exit(None)

    def observe(self, func):
        self._do_trace_code(func.__code__)
        return func

    def print_graph(self):
        """
        Prints the trace spans in a tree-like indented format.
        """

        print("-" * 80)
        parents = [span for span in self._trace_spans if span.parent_span_id is None]
        children = [
            span for span in self._trace_spans if span.parent_span_id is not None
        ]

        INDENT_SIZE = 3

        def print_span(span: CurrentSpanExit, depth: int = 0):
            indent = "  " * depth * INDENT_SIZE

            args_str = ""
            if span.inputs:
                args_list = []
                for k, v in span.inputs.items():
                    args_list.append(f"{k}={v!r}")
                args_str = ", ".join(args_list)

            output_str = f"{span.output!r}"

            duration_ms = (span.end_time - span.start_time) * 1000
            error_str = ""
            if span.error is not None and isinstance(span.error, dict):
                error_type = span.error.get("type", "UnknownExceptionType")
                error_message = span.error.get("message", "No exception message")
                error_str = f" [ERROR: {error_type}: {error_message}]"
            print(
                f"{indent}→ {span.name}({args_str}) => {output_str} "
                f"[{duration_ms:.2f} ms]{error_str}"
            )
            for child in children:
                if child.parent_span_id == span.span_id:
                    print_span(child, depth + 1)

        for root_span in parents:
            print_span(root_span)
            print()

    def _flush_spans(self):
        """
        Flush the trace spans to the storage client.
        This will save the trace spans to the configured storage client.
        """
        if len(self._trace_spans) == 0:
            logging.info("No trace spans to flush.")
            return

        logging.info(f"Flushing {len(self._trace_spans)} trace spans to storage.")

        depth_map = get_span_exit_depth_map(self._trace_spans)
        trace_spans = cast(List[TraceSpan], [])
        for span in self._trace_spans:
            trace_spans.append(
                TraceSpan(
                    span_id=span.span_id,
                    trace_id=self.trace_id,
                    function=span.name,
                    depth=depth_map.get(self._trace_spans[0].span_id, -1),
                    span_type=span.span_type,
                    parent_span_id=span.parent_span_id,
                    duration=(span.end_time - span.start_time),
                    inputs=span.inputs or {},
                    output=span.output,
                    error=span.error,
                )
            )

        trace_save = TraceSave(
            trace_id=self.trace_id,
            name=self.name,
            created_at=datetime.utcfromtimestamp(time.time()).isoformat(),
            duration=(self._trace_spans[-1].end_time - self._trace_spans[0].start_time),
            trace_spans=trace_spans,
            project_name=self.name,
            evaluation_runs=[],
        )

        self._storage_client.save_trace(
            trace_data=trace_save, trace_id=self.trace_id, project_name=self.name
        )

        self.print_graph()
        self._trace_spans.clear()

    @contextmanager
    def daemon(self, deep_tracing: bool = False):
        """
        Install the trace daemon.
        """
        try:
            sys.settrace(self._trace_daemon_factory_(deep_tracing))
            yield
        finally:
            sys.settrace(None)
            self._flush_spans()
