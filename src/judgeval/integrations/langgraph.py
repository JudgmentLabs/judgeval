from typing import Any, Dict, List, Optional, Sequence, Callable, TypedDict
from uuid import UUID
import time
import uuid
import traceback # For detailed error logging
import contextvars # <--- Import contextvars
from dataclasses import dataclass

from judgeval.common.tracer import TraceClient, TraceEntry, Tracer, SpanType, EvaluationConfig
from judgeval.data import Example # Import Example
from judgeval.scorers import AnswerRelevancyScorer, JudgevalScorer, APIJudgmentScorer # Import Scorer and base scorer types

from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.documents import Document

# --- Get context vars from tracer module ---
# Assuming tracer.py defines these and they are accessible
# If not, redefine them here or adjust import
from judgeval.common.tracer import current_span_var, current_trace_var # <-- Import current_trace_var

# --- Constants for Logging ---
HANDLER_LOG_PREFIX = "[JudgevalHandlerLog]"

# --- NEW __init__ ---
class JudgevalCallbackHandler(BaseCallbackHandler):
    """
    LangChain Callback Handler using run_id/parent_run_id for hierarchy.
    Manages its own internal TraceClient instance created upon first use.
    Includes verbose logging and defensive checks.
    """
    # Make all properties ignored by LangChain's callback system
    # to prevent unexpected serialization issues.
    lc_serializable = False
    lc_kwargs = {}

    # --- NEW __init__ ---
    def __init__(self, tracer: Tracer):
        # --- Enhanced Logging ---
        # instance_id = id(self)
        # # print(f"{HANDLER_LOG_PREFIX} *** Handler instance {instance_id} __init__ called. ***")
        # --- End Enhanced Logging ---
        self.tracer = tracer
        self._trace_client: Optional[TraceClient] = None
        self._run_id_to_span_id: Dict[UUID, str] = {}
        self._span_id_to_start_time: Dict[str, float] = {}
        self._span_id_to_depth: Dict[str, int] = {}
        self._run_id_to_context_token: Dict[UUID, contextvars.Token] = {}
        self._root_run_id: Optional[UUID] = None
        self._trace_saved: bool = False # Flag to prevent actions after trace is saved
        self._run_id_to_start_inputs: Dict[UUID, Dict] = {} # <<< ADDED input storage

        # --- Token Count Accumulators ---
        # self._current_prompt_tokens = 0
        # self._current_completion_tokens = 0
        # --- End Token Count Accumulators ---

        self.executed_nodes: List[str] = []
        self.executed_tools: List[str] = []
        self.executed_node_tools: List[str] = []
        self.traces: List[Dict[str, Any]] = []
    # --- END NEW __init__ ---

    # --- MODIFIED _ensure_trace_client ---
    def _ensure_trace_client(self, run_id: UUID, parent_run_id: Optional[UUID], event_name: str) -> Optional[TraceClient]:
        """
        Ensures the internal trace client is initialized, creating it only once
        per handler instance lifecycle (effectively per graph invocation).
        Returns the client or None.
        """
        # handler_instance_id = id(self)
        # log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"

        # If trace already saved, do nothing.
        # if self._trace_saved:
        #     # print(f"{log_prefix} Trace already saved. Skipping client check for {event_name} ({run_id}).")
        #     return None

        # If a client already exists, return it.
        if self._trace_client:
            # # print(f"{log_prefix} Reusing existing TraceClient (ID: {self._trace_client.trace_id}) for {event_name} ({run_id}).")
            return self._trace_client

        # If no client exists, initialize it NOW.
        # # print(f"{log_prefix} No TraceClient exists. Initializing for first event: {event_name} ({run_id})...")
        trace_id = str(uuid.uuid4())
        project = self.tracer.project_name
        try:
            # Use event_name as the initial trace name, might be updated later by on_chain_start if root
            client_instance = TraceClient(
                self.tracer, trace_id, event_name, project_name=project,
                overwrite=False, rules=self.tracer.rules,
                enable_monitoring=self.tracer.enable_monitoring,
                enable_evaluations=self.tracer.enable_evaluations
            )
            self._trace_client = client_instance
            if self._trace_client:
                self._root_run_id = run_id # Assign the first run_id encountered as the tentative root
                self._trace_saved = False # Ensure flag is reset
                # # print(f"{log_prefix} Initialized NEW TraceClient: ID={self._trace_client.trace_id}, InitialName='{event_name}', Root Run ID={self._root_run_id}")
                # Set active client on Tracer (important for potential fallbacks)
                self.tracer._active_trace_client = self._trace_client
                return self._trace_client
            else:
                # # print(f"{log_prefix} FATAL: TraceClient creation failed unexpectedly for {event_name} ({run_id}).")
                return None
        except Exception as e:
            # # print(f"{log_prefix} FATAL: Exception initializing TraceClient for {event_name} ({run_id}): {e}")
            # # print(traceback.format_exc())
            self._trace_client = None
            self._root_run_id = None
            return None
    # --- END MODIFIED _ensure_trace_client ---

    def _log(self, message: str):
        """Helper for consistent logging format."""
        pass

    def _start_span_tracking(
        self,
        trace_client: TraceClient, # Expect a valid client
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        span_type: SpanType = "span",
        inputs: Optional[Dict[str, Any]] = None
    ):
        # self._log(f"_start_span_tracking called for: name='{name}', run_id={run_id}, parent_run_id={parent_run_id}, span_type={span_type}")

        # --- Add explicit check for trace_client ---
        if not trace_client:
            # --- Enhanced Logging ---
            # handler_instance_id = id(self)
            # log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
            # --- End Enhanced Logging ---
            # self._log(f"{log_prefix} FATAL ERROR in _start_span_tracking: trace_client argument is None for name='{name}', run_id={run_id}. Aborting span start.")
            return
        # --- End check ---
        # --- Enhanced Logging ---
        # handler_instance_id = id(self)
        # log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # trace_client_instance_id = id(trace_client) if trace_client else 'None'
        # # print(f"{log_prefix} _start_span_tracking: Using TraceClient ID: {trace_client_instance_id}")
        # --- End Enhanced Logging ---

        start_time = time.time()
        span_id = str(uuid.uuid4())
        parent_span_id: Optional[str] = None
        current_depth = 0

        if parent_run_id and parent_run_id in self._run_id_to_span_id:
            parent_span_id = self._run_id_to_span_id[parent_run_id]
            if parent_span_id in self._span_id_to_depth:
                parent_depth = self._span_id_to_depth[parent_span_id]
                current_depth = parent_depth + 1
                # self._log(f"  Found parent span_id={parent_span_id} with depth={parent_depth}. New depth={current_depth}.")
            else:
                # self._log(f"  WARNING: Parent span depth not found for parent_span_id: {parent_span_id}. Setting depth to 0.")
                current_depth = 0
        elif parent_run_id:
            # self._log(f"  WARNING: parent_run_id {parent_run_id} provided for '{name}' ({run_id}) but parent span not tracked. Treating as depth 0.")
            pass
        else:
            # self._log(f"  No parent_run_id provided. Treating '{name}' as depth 0.")
            pass

        self._run_id_to_span_id[run_id] = span_id
        self._span_id_to_start_time[span_id] = start_time
        self._span_id_to_depth[span_id] = current_depth
        # self._log(f"  Tracking new span: span_id={span_id}, depth={current_depth}")

        try:
            trace_client.add_entry(TraceEntry(
                type="enter", span_id=span_id, trace_id=trace_client.trace_id,
                parent_span_id=parent_span_id, function=name, depth=current_depth,
                message=name, created_at=start_time, span_type=span_type
            ))
            # self._log(f"  Added 'enter' entry for span_id={span_id}")
        except Exception as e:
            # self._log(f"  ERROR adding 'enter' entry for span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

        if inputs:
            # Pass the already validated trace_client
            self._record_input_data(trace_client, run_id, inputs)

        # --- Set SPAN context variable ONLY for chain (node) spans (Sync version) ---
        if span_type == "chain":
            try:
                token = current_span_var.set(span_id)
                self._run_id_to_context_token[run_id] = token
                # self._log(f"  Set current_span_var to {span_id} for run_id {run_id} (type: chain) in Sync Handler")
            except Exception as e:
                # self._log(f"  ERROR setting current_span_var for run_id {run_id} in Sync Handler: {e}")
                pass
        # --- END ---

        try:
            # TODO: Check if trace_client.add_entry needs await if TraceClient becomes async
            trace_client.add_entry(TraceEntry(
                type="enter", span_id=span_id, trace_id=trace_client.trace_id,
                parent_span_id=parent_span_id, function=name, depth=current_depth,
                message=name, created_at=start_time, span_type=span_type
            ))
            # self._log(f"  Added 'enter' entry for span_id={span_id}")
        except Exception as e:
            # self._log(f"  ERROR adding 'enter' entry for span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

        if inputs:
            # _record_input_data is also sync for now
            self._record_input_data(trace_client, run_id, inputs)

    # --- NEW _end_span_tracking ---
    def _end_span_tracking(
        self,
        trace_client: TraceClient, # Expect a valid client
        run_id: UUID,
        span_type: SpanType = "span",
        outputs: Optional[Any] = None,
        error: Optional[BaseException] = None
    ):
        # self._log(f"_end_span_tracking called for: run_id={run_id}, span_type={span_type}")

        # --- Define instance_id early for logging/cleanup ---
        instance_id = id(self)

        if not trace_client:
            # Use instance_id defined above
            # log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {instance_id}]"
            # self._log(f"{log_prefix} FATAL ERROR in _end_span_tracking: trace_client argument is None for run_id={run_id}. Aborting span end.")
            return

        # Use instance_id defined above
        # log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {instance_id}]"
        # trace_client_instance_id = id(trace_client) if trace_client else 'None'
        # # print(f"{log_prefix} _end_span_tracking: Using TraceClient ID: {trace_client_instance_id}")

        if run_id not in self._run_id_to_span_id:
            # self._log(f"  WARNING: Attempting to end span for untracked run_id: {run_id}")
            # Allow root run end to proceed for cleanup/save attempt even if span wasn't tracked
            if run_id != self._root_run_id:
                 return
            else:
                 # self._log(f"  Allowing root run {run_id} end logic to proceed despite untracked span.")
                 span_id = None # Indicate span wasn't found for duration/metadata lookup
        else:
            span_id = self._run_id_to_span_id[run_id]

        start_time = self._span_id_to_start_time.get(span_id) if span_id else None
        depth = self._span_id_to_depth.get(span_id, 0) if span_id else 0 # Use 0 depth if span_id is None
        duration = time.time() - start_time if start_time is not None else None
        # self._log(f"  Ending span for run_id={run_id} (span_id={span_id}). Start time={start_time}, Duration={duration}, Depth={depth}")

        # Record output/error first
        if error:
            # self._log(f"  Recording error for run_id={run_id} (span_id={span_id}): {error}")
            self._record_output_data(trace_client, run_id, error)
        elif outputs is not None:
            # output_repr = repr(outputs)
            # log_output = (output_repr[:100] + '...') if len(output_repr) > 103 else output_repr
            # self._log(f"  Recording output for run_id={run_id} (span_id={span_id}): {log_output}")
            self._record_output_data(trace_client, run_id, outputs)

        # Add exit entry (only if span was tracked)
        if span_id:
            entry_function_name = "unknown"
            try:
                if hasattr(trace_client, 'entries') and trace_client.entries:
                    entry_function_name = next((e.function for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), "unknown")
                else:
                    # self._log(f"  WARNING: Cannot determine function name for exit span_id {span_id}, trace_client.entries missing or empty.")
                    pass
            except Exception as e:
                # self._log(f"  ERROR finding function name for exit entry span_id {span_id}: {e}")
                # # print(traceback.format_exc())
                pass

            try:
                trace_client.add_entry(TraceEntry(
                    type="exit", span_id=span_id, trace_id=trace_client.trace_id,
                    depth=depth, created_at=time.time(), duration=duration,
                    span_type=span_type, function=entry_function_name
                ))
                # self._log(f"  Added 'exit' entry for span_id={span_id}, function='{entry_function_name}'")
            except Exception as e:
                # self._log(f"  ERROR adding 'exit' entry for span_id {span_id}: {e}")
                # # print(traceback.format_exc())
                pass

            # Clean up dictionaries for this specific span
            if span_id in self._span_id_to_start_time: del self._span_id_to_start_time[span_id]
            if span_id in self._span_id_to_depth: del self._span_id_to_depth[span_id]

            # Pop context token (Sync version) but don't reset
            token = self._run_id_to_context_token.pop(run_id, None)
            if token:
                # self._log(f" Popped token for run_id {run_id} (was {span_id}), not resetting context var.")
                pass
        else:
             # self._log(f"  Skipping exit entry and cleanup for run_id {run_id} as span_id was not found.")
             pass

        # Check if this is the root run ending
        if run_id == self._root_run_id:
            trace_saved_successfully = False # Track save success
            try:
                # --- Aggregate and Set Token Counts BEFORE Saving ---
                # if self._trace_client and not self._trace_saved:
                #     total_tokens = self._current_prompt_tokens + self._current_completion_tokens
                #     aggregated_token_counts = {
                #         'prompt_tokens': self._current_prompt_tokens,
                #         'completion_tokens': self._current_completion_tokens,
                #         'total_tokens': total_tokens
                #     }
                #     # Assuming TraceClient has an attribute to hold the trace data being built
                #     try:
                #          # Attempt to set the attribute directly
                #          # Check if the attribute exists and is meant for this purpose
                #          if hasattr(self._trace_client, 'token_counts'):
                #              self._trace_client.token_counts = aggregated_token_counts
                #              self._log(f"Set aggregated token_counts on TraceClient for trace {self._trace_client.trace_id}: {aggregated_token_counts}")
                #          else:
                #              # If the attribute doesn't exist, maybe update the trace data dict directly if possible?
                #              # This part is speculative without knowing TraceClient internals.
                #              # E.g., if trace_client has a `_trace_data` dict:
                #              # if hasattr(self._trace_client, '_trace_data') and isinstance(self._trace_client._trace_data, dict):
                #              #     self._trace_client._trace_data['token_counts'] = aggregated_token_counts
                #              #     self._log(f"Updated _trace_data['token_counts'] on TraceClient for trace {self._trace_client.trace_id}: {aggregated_token_counts}")
                #              # else:
                #                 self._log(f"WARNING: Could not set 'token_counts' on TraceClient for trace {self._trace_client.trace_id}. Aggregated counts might be lost.")
                #     except Exception as set_tc_e:
                #          self._log(f"ERROR setting token_counts on TraceClient for trace {self._trace_client.trace_id}: {set_tc_e}")
                # --- End Token Count Aggregation ---

                # Reset root run id after attempt
                self._root_run_id = None
                # Reset input storage for this handler instance
                self._run_id_to_start_inputs = {}
                self._log(f"Reset root run ID and input storage for handler {instance_id}.")

                self._log(f"Root run {run_id} finished. Attempting to save trace...")
                if self._trace_client and not self._trace_saved: # Check if not already saved
                    try:
                        # TODO: Check if trace_client.save needs await if TraceClient becomes async
                        trace_id, trace_data = self._trace_client.save(overwrite=self._trace_client.overwrite) # Use client's overwrite setting
                        self.traces.append(trace_data)
                        self._log(f"Trace {trace_id} successfully saved.")
                        self._trace_saved = True # Set flag only after successful save
                        trace_saved_successfully = True # Mark success
                    except Exception as e:
                        self._log(f"ERROR saving trace {self._trace_client.trace_id}: {e}")
                        # print(traceback.format_exc())
                    # REMOVED FINALLY BLOCK THAT RESET STATE HERE
                elif self._trace_client and self._trace_saved:
                     self._log(f"  Trace {self._trace_client.trace_id} already saved. Skipping save.")
                else:
                    self._log(f"  WARNING: Root run {run_id} ended, but trace client was None. Cannot save trace.")
            finally:
                # --- NEW: Consolidated Cleanup Logic --- 
                # This block executes regardless of save success/failure
                self._log(f"  Performing cleanup for root run {run_id} in handler {instance_id}.")
                # Reset root run id
                self._root_run_id = None
                # Reset input storage for this handler instance
                self._run_id_to_start_inputs = {}
                # --- Reset Token Counters ---
                # self._current_prompt_tokens = 0
                # self._current_completion_tokens = 0
                # self._log("  Reset token counters.")
                # --- End Reset Token Counters ---
                # Reset tracer's active client ONLY IF it was this handler's client
                if self.tracer._active_trace_client == self._trace_client:
                    self.tracer._active_trace_client = None
                    self._log("  Reset active_trace_client on Tracer.")
                # Completely remove trace_context_token cleanup as it's not used in sync handler
                # Optionally: Reset the entire trace client instance for this handler?
                # self._trace_client = None # Uncomment if handler should reset client completely after root run
                self._log(f"  Cleanup complete for root run {run_id}.")
                # --- End Cleanup Logic ---

    def _record_input_data(self,
                           trace_client: TraceClient,
                           run_id: UUID,
                           inputs: Dict[str, Any]):
        # self._log(f"_record_input_data called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            # self._log(f"  WARNING: Attempting to record input for untracked run_id: {run_id}")
            return
        if not trace_client:
             # self._log(f"  ERROR: TraceClient is None when trying to record input for run_id={run_id}")
             return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        # self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            # Find the corresponding 'enter' entry to get the function name and span type
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                # self._log(f"  Found function='{function_name}', span_type='{span_type}' for input span_id={span_id}")
            else:
                # self._log(f"  WARNING: Could not find 'enter' entry for input span_id={span_id}")
                pass
        except Exception as e:
            # self._log(f"  ERROR finding enter entry for input span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

        try:
            input_entry = TraceEntry(
                type="input",
                span_id=span_id,
                trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None), # Get parent from enter entry
                function=function_name,
                depth=depth,
                message=f"Input to {function_name}",
                created_at=time.time(),
                inputs=inputs,
                span_type=span_type
            )
            trace_client.add_entry(input_entry)
            # self._log(f"  Added 'input' entry directly for span_id={span_id}")
        except Exception as e:
            # self._log(f"  ERROR adding 'input' entry directly for span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

    def _record_output_data(self,
                            trace_client: TraceClient,
                            run_id: UUID,
                            output: Any):
        # self._log(f"_record_output_data called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            # self._log(f"  WARNING: Attempting to record output for untracked run_id: {run_id}")
            return
        if not trace_client:
            # self._log(f"  ERROR: TraceClient is None when trying to record output for run_id={run_id}")
            return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        # self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            # Find the corresponding 'enter' entry to get the function name and span type
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                # self._log(f"  Found function='{function_name}', span_type='{span_type}' for output span_id={span_id}")
            else:
                 # self._log(f"  WARNING: Could not find 'enter' entry for output span_id={span_id}")
                 pass
        except Exception as e:
            # self._log(f"  ERROR finding enter entry for output span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

        try:
            output_entry = TraceEntry(
                type="output",
                span_id=span_id,
                trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None), # Get parent from enter entry
                function=function_name,
                depth=depth,
                message=f"Output from {function_name}",
                created_at=time.time(),
                output=output, # Langchain outputs are typically serializable directly
                span_type=span_type
            )
            trace_client.add_entry(output_entry)
            self._log(f"  Added 'output' entry directly for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'output' entry directly for span_id {span_id}: {e}")
            # print(traceback.format_exc())

    def _record_error(self,
                      trace_client: TraceClient,
                      run_id: UUID,
                      error: Any):
        # self._log(f"_record_error called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            # self._log(f"  WARNING: Attempting to record error for untracked run_id: {run_id}")
            return
        if not trace_client:
            # self._log(f"  ERROR: TraceClient is None when trying to record error for run_id={run_id}")
            return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        # self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            # Find the corresponding 'enter' entry to get the function name and span type
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                # self._log(f"  Found function='{function_name}', span_type='{span_type}' for error span_id={span_id}")
            else:
                # self._log(f"  WARNING: Could not find 'enter' entry for error span_id={span_id}")
                pass
        except Exception as e:
            # self._log(f"  ERROR finding enter entry for error span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

        try:
            error_entry = TraceEntry(
                type="error",
                span_id=span_id,
                trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None), # Get parent from enter entry
                function=function_name,
                depth=depth,
                message=f"Error in {function_name}",
                created_at=time.time(),
                error=str(error), # Convert error to string for serialization
                span_type=span_type
            )
            trace_client.add_entry(error_entry)
            # self._log(f"  Added 'error' entry directly for span_id={span_id}")
        except Exception as e:
            # self._log(f"  ERROR adding 'error' entry directly for span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

    # --- Callback Methods ---
    # Each method now ensures the trace client exists before proceeding

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        serialized_name = serialized.get('name', 'Unknown') if serialized else "Unknown (Serialized=None)"
        # print(f"{log_prefix} ENTERING on_retriever_start: name='{serialized_name}', run_id={run_id}. Parent: {parent_run_id}")

        try:
            name = f"RETRIEVER_{(serialized_name).upper()}"
            # Pass parent_run_id to _ensure_trace_client
            trace_client = self._ensure_trace_client(run_id, parent_run_id, name) # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_retriever_start for {run_id}.")
                return

            inputs = {'query': query, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type="retriever", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_retriever_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # print(f"{log_prefix} ENTERING on_retriever_end: run_id={run_id}. Parent: {parent_run_id}")

        try:
            # Pass parent_run_id to _ensure_trace_client (though less critical on end events)
            trace_client = self._ensure_trace_client(run_id, parent_run_id, "RetrieverEnd") # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_retriever_end for {run_id}.")
                return

            doc_summary = [{"index": i, "page_content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(documents)]
            outputs = {"document_count": len(documents), "documents": doc_summary, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="retriever", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_retriever_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        serialized_name = serialized.get('name') if serialized else "Unknown (Serialized=None)"
        # print(f"{log_prefix} ENTERING on_chain_start: name='{serialized_name}', run_id={run_id}. Parent: {parent_run_id}")

        # --- Determine Name and Span Type ---
        span_type: SpanType = "chain"
        name = serialized_name if serialized_name else "Unknown Chain" # Default name
        node_name = metadata.get("langgraph_node") if metadata else None
        is_langgraph_root_kwarg = kwargs.get('name') == 'LangGraph' # Check kwargs for explicit root name
        # More robust root detection: Often the first chain event with parent_run_id=None *is* the root.
        is_potential_root_event = parent_run_id is None

        if 'langsmith:hidden' in tags:
            pass

        if node_name:
            name = node_name # Use node name if available
            self._log(f"  LangGraph Node Start: '{name}', run_id={run_id}")
            if name not in self.executed_nodes: self.executed_nodes.append(name)
        elif is_langgraph_root_kwarg and is_potential_root_event:
             name = "LangGraph" # Explicit root detected
             self._log(f"  LangGraph Root Start Detected (kwargs): run_id={run_id}")
        # Add handling for other potential LangChain internal chains if needed, e.g., "RunnableSequence"

        # --- Ensure Trace Client ---
        try:
            # Pass parent_run_id to _ensure_trace_client
            trace_client = self._ensure_trace_client(run_id, parent_run_id, name) # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_chain_start for {run_id} ('{name}').")
                return

            # --- Update Trace Name if Root ---
            # If this is the root event (parent_run_id is None) and the trace client was just created,
            # ensure the trace name reflects the graph's name ('LangGraph' usually).
            if is_potential_root_event and run_id == self._root_run_id and trace_client.name != name:
                 self._log(f"  Updating trace name from '{trace_client.name}' to '{name}' for root run {run_id}")
                 trace_client.name = name # Update trace name to the determined root name

            # --- Start Span Tracking ---
            combined_inputs = {'inputs': inputs, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type=span_type, inputs=inputs)
            # --- Store inputs for potential evaluation later --- 
            self._run_id_to_start_inputs[run_id] = inputs # Store the raw inputs dict
            self._log(f"  Stored inputs for run_id {run_id}")
            # --- End Store inputs --- 

        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())


    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # print(f"{log_prefix} ENTERING on_chain_end: run_id={run_id}. Parent: {parent_run_id}")

        # --- Define instance_id for logging --- 
        instance_id = handler_instance_id # Use the already obtained id

        if 'langsmith:hidden' in tags:
            pass

        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, "ChainEnd") # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_chain_end for {run_id}.")
                return

            span_id = self._run_id_to_span_id.get(run_id)
            span_type: SpanType = "chain" # Default
            if span_id:
                try:
                    if hasattr(trace_client, 'entries') and trace_client.entries:
                        enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                        if enter_entry: span_type = enter_entry.span_type
                    else: self._log(f"  WARNING: trace_client.entries empty/missing for on_chain_end span_id={span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_end: {e}")
            else:
                 self._log(f"  WARNING: No span_id found for run_id {run_id} in on_chain_end.")
                 # If it's the root run ending, _end_span_tracking will handle cleanup/save
                 if run_id == self._root_run_id:
                      self._log(f"  Run ID {run_id} matches root. Proceeding to _end_span_tracking for potential save.")
                 else:
                      return # Don't call end tracking if it's not the root and span wasn't tracked

            # --- Store input in on_chain_start if needed for evaluation --- 
            # Retrieve stored inputs
            start_inputs = self._run_id_to_start_inputs.get(run_id, {})
            # TODO: Determine how to reliably extract the original user prompt from start_inputs
            # For the demo, the 'generate_recommendations' node receives the full state, not the initial prompt.
            # Using a placeholder for now.
            user_prompt_for_eval = "Unknown Input" # Placeholder - Needs refinement based on graph structure
            # user_prompt_for_eval = start_inputs.get("messages", [{}])[-1].get("content", "Unknown Input") # Example: If input has messages list

            # --- Trigger evaluation --- 
            if "recommendations" in outputs and span_id: # Ensure span_id exists
                self._log(f"[Async Handler {instance_id}] Chain end for run_id {run_id} (span_id={span_id}) identified as recommendation node. Attempting evaluation.")
                recommendations_output = outputs.get("recommendations")

                if recommendations_output:
                    eval_example = Example(
                        input=user_prompt_for_eval, # Requires modification to store/retrieve this
                        actual_output=recommendations_output
                    )
                    # TODO: Get model name dynamically if possible
                    model_name = "gpt-4" # Placeholder

                    self._log(f"[Async Handler {instance_id}] Submitting evaluation for span_id={span_id}")
                    try:
                        # Call evaluate on the trace client, passing the specific span_id
                        # The TraceClient.async_evaluate now accepts and prioritizes this span_id.
                        trace_client.async_evaluate(
                            scorers=[AnswerRelevancyScorer(threshold=0.5)], # Ensure this scorer is imported
                            example=eval_example,
                            model=model_name,
                            span_id=span_id # Pass the specific span_id for this node run
                        )
                        self._log(f"[Async Handler {instance_id}] Evaluation submitted successfully for span_id={span_id}.")
                    except Exception as eval_e:
                        self._log(f"[Async Handler {instance_id}] ERROR submitting evaluation for span_id={span_id}: {eval_e}")
                        # print(traceback.format_exc()) # Print traceback for evaluation errors
                else:
                    self._log(f"[Async Handler {instance_id}] Skipping evaluation for run_id {run_id} (span_id={span_id}): Missing recommendations output.")
            elif "recommendations" in outputs:
                 self._log(f"[Async Handler {instance_id}] Skipping evaluation for run_id {run_id}: Span ID not found.")

            # --- Existing span ending logic --- 
            # Determine span_type for end_span_tracking (copied from sync handler)
            end_span_type: SpanType = "chain" # Default
            if span_id: # Check if span_id was actually found
                try:
                    if hasattr(trace_client, 'entries') and trace_client.entries:
                        enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                        if enter_entry: end_span_type = enter_entry.span_type
                    else: self._log(f"  WARNING: trace_client.entries empty/missing for on_chain_end span_id={span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_end: {e}")
            else:
                 self._log(f"  WARNING: No span_id found for run_id {run_id} in on_chain_end, using default span_type='chain'.")

            # Prepare outputs for end tracking (moved down)
            combined_outputs = {"outputs": outputs, "tags": tags, "kwargs": kwargs}

            # Call end_span_tracking with potentially determined span_type
            self._end_span_tracking(trace_client, run_id, span_type=end_span_type, outputs=combined_outputs)

            # --- Root node cleanup (Existing logic - slightly modified save call) ---
            if run_id == self._root_run_id:
                self._log(f"Root run {run_id} finished. Attempting to save trace...")
                if trace_client and not self._trace_saved:
                    try:
                        # Save might need to be async if TraceClient methods become async
                        # Pass overwrite=True based on client's setting
                        trace_id_saved, trace_data = trace_client.save(overwrite=trace_client.overwrite)
                        self.traces.append(trace_data)
                        self._trace_saved = True
                        self._log(f"Trace {trace_id_saved} successfully saved.")
                        # Reset tracer's active client *after* successful save
                        if self.tracer._active_trace_client == trace_client:
                            self.tracer._active_trace_client = None 
                            self._log("Reset active_trace_client on Tracer.")
                    except Exception as e:
                        self._log(f"ERROR saving trace {trace_client.trace_id}: {e}")
                        # print(traceback.format_exc())
                elif trace_client and self._trace_saved:
                    self._log(f"Trace {trace_client.trace_id} already saved. Skipping save for root run {run_id}.")
                elif not trace_client:
                    self._log(f"Skipping trace save for root run {run_id}: No client available.")
                
                # Reset root run id after attempt
                self._root_run_id = None
                # Reset input storage for this handler instance
                self._run_id_to_start_inputs = {} 
                self._log(f"Reset root run ID and input storage for handler {instance_id}.")

            # --- SYNC: Attempt Evaluation by checking output metadata --- 
            eval_config: Optional[EvaluationConfig] = None
            node_name = "unknown_node" # Default node name
            # Ensure trace_client exists before proceeding with eval logic that uses it
            if trace_client:
                if span_id: # Try to find the node name from the 'enter' entry
                    try:
                        if hasattr(trace_client, 'entries') and trace_client.entries:
                            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                            if enter_entry: node_name = enter_entry.function
                    except Exception as e:
                        self._log(f"  ERROR finding node name for span_id {span_id} in on_chain_end: {e}")
                
                if span_id and "_judgeval_eval" in outputs: # Only attempt if span exists and key is present
                    raw_eval_config = outputs.get("_judgeval_eval")
                    if isinstance(raw_eval_config, EvaluationConfig):
                        eval_config = raw_eval_config
                        self._log(f"{log_prefix} Found valid EvaluationConfig in outputs for node='{node_name}'.")
                    elif isinstance(raw_eval_config, dict):
                         # Attempt to reconstruct from dict
                         try:
                             if "scorers" in raw_eval_config and "example" in raw_eval_config:
                                 example_data = raw_eval_config["example"]
                                 reconstructed_example = Example(**example_data) if isinstance(example_data, dict) else example_data

                                 if isinstance(reconstructed_example, Example):
                                     eval_config = EvaluationConfig(
                                         scorers=raw_eval_config["scorers"], 
                                         example=reconstructed_example,
                                         model=raw_eval_config.get("model"),
                                         log_results=raw_eval_config.get("log_results", True)
                                     )
                                     self._log(f"{log_prefix} Reconstructed EvaluationConfig from dict in outputs for node='{node_name}'.")
                                 else:
                                    self._log(f"{log_prefix} Could not reconstruct Example from dict in _judgeval_eval for node='{node_name}'. Skipping evaluation.")
                             else:
                                 self._log(f"{log_prefix} Dict in _judgeval_eval missing required keys ('scorers', 'example') for node='{node_name}'. Skipping evaluation.")
                         except Exception as recon_e:
                             self._log(f"{log_prefix} ERROR attempting to reconstruct EvaluationConfig from dict for node='{node_name}': {recon_e}")
                             # print(traceback.format_exc()) # Print traceback for reconstruction errors
                    else:
                        self._log(f"{log_prefix} Found '_judgeval_eval' key in outputs for node='{node_name}', but it wasn't an EvaluationConfig object or reconstructable dict. Skipping evaluation.")

                # Check eval_config *and* span_id again (this should be indented correctly)
                if eval_config and span_id: 
                    self._log(f"{log_prefix} Submitting evaluation for span_id={span_id}")
                    try:
                        # Ensure example has trace_id set if not already present
                        if not hasattr(eval_config.example, 'trace_id') or not eval_config.example.trace_id:
                            # Use the correct variable name 'trace_client' here
                            eval_config.example.trace_id = trace_client.trace_id 
                            self._log(f"{log_prefix} Set trace_id={trace_client.trace_id} on evaluation example.")
                        
                        # Call async_evaluate on the TraceClient instance ('trace_client')
                        # Use the correct variable name 'trace_client' here
                        trace_client.async_evaluate( # <-- Fix: Use trace_client
                            scorers=eval_config.scorers,
                            example=eval_config.example,
                            model=eval_config.model,
                            log_results=eval_config.log_results,
                            span_id=span_id # Pass the specific span_id for this node run
                        )
                        self._log(f"{log_prefix} Evaluation submitted successfully for span_id={span_id}.")
                    except Exception as eval_e:
                        self._log(f"{log_prefix} ERROR submitting evaluation for span_id={span_id}: {eval_e}")
                        # print(traceback.format_exc()) # Print traceback for evaluation errors
                elif "_judgeval_eval" in outputs and not span_id:
                     self._log(f"{log_prefix} WARNING: Found _judgeval_eval in outputs, but span_id for run_id {run_id} was not found. Cannot submit evaluation.")
            # --- End SYNC Evaluation Logic ---

        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # print(f"{log_prefix} ENTERING on_chain_error: run_id={run_id}. Parent: {parent_run_id}")

        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, "ChainError") # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_chain_error for {run_id}.")
                return

            span_id = self._run_id_to_span_id.get(run_id)
            span_type: SpanType = "chain" # Default
            if span_id:
                try:
                     if hasattr(trace_client, 'entries') and trace_client.entries:
                        enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                        if enter_entry: span_type = enter_entry.span_type
                     else: self._log(f"  WARNING: trace_client.entries empty/missing for on_chain_error span_id={span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_error: {e}")
            else:
                 self._log(f"  WARNING: No span_id found for run_id {run_id} in on_chain_error.")
                 # Let _end_span_tracking handle potential root run cleanup
                 if run_id != self._root_run_id:
                      return

            self._end_span_tracking(trace_client, run_id, span_type=span_type, error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        name = serialized.get("name", "Unnamed Tool") if serialized else "Unknown Tool (Serialized=None)"
        # print(f"{log_prefix} ENTERING on_tool_start: name='{name}', run_id={run_id}. Parent: {parent_run_id}")

        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, name) # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_tool_start for {run_id}.")
                return

            combined_inputs = {'input_str': input_str, 'inputs': inputs, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type="tool", inputs=inputs)

            # --- Track executed tools (remains the same) ---
            if name not in self.executed_tools: self.executed_tools.append(name)
            parent_node_name = None
            if parent_run_id and parent_run_id in self._run_id_to_span_id:
                parent_span_id = self._run_id_to_span_id[parent_run_id]
                try:
                    if hasattr(trace_client, 'entries') and trace_client.entries:
                        parent_enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == parent_span_id and e.type == "enter" and e.span_type == "chain"), None)
                        if parent_enter_entry:
                            parent_node_name = parent_enter_entry.function
                    else: self._log(f"  WARNING: trace_client.entries missing for tool start parent {parent_span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding parent node name for tool start span_id {parent_span_id}: {e}")

            node_tool = f"{parent_node_name}:{name}" if parent_node_name else name
            if node_tool not in self.executed_node_tools: self.executed_node_tools.append(node_tool)
            self._log(f"  Tracked node_tool: {node_tool}")
            # --- End Track executed tools ---
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())


    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # print(f"{log_prefix} ENTERING on_tool_end: run_id={run_id}. Parent: {parent_run_id}")

        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, "ToolEnd") # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_tool_end for {run_id}.")
                return
            outputs = {"output": output, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="tool", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # print(f"{log_prefix} ENTERING on_tool_error: run_id={run_id}. Parent: {parent_run_id}")

        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, "ToolError") # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_tool_error for {run_id}.")
                return
            self._end_span_tracking(trace_client, run_id, span_type="tool", error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, invocation_params: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        llm_name = name or serialized.get("name", "LLM Call")
        # print(f"{log_prefix} ENTERING on_llm_start: name='{llm_name}', run_id={run_id}. Parent: {parent_run_id}")

        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, llm_name) # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_llm_start for {run_id}.")
                return
            inputs = {'prompts': prompts, 'invocation_params': invocation_params or kwargs, 'options': options, 'tags': tags, 'metadata': metadata, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, llm_name, span_type="llm", inputs=prompts)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # print(f"{log_prefix} ENTERING on_llm_end: run_id={run_id}. Parent: {parent_run_id}")

        # --- Debugging unchanged ---
        # # print(f"{log_prefix} [DEBUG on_llm_end] Received response object for run_id={run_id}:")
        # try:
        #     from rich import print as rprint
        #     r# print(response)
        # except ImportError: # print(response)
        # # print(f"{log_prefix} [DEBUG on_llm_end] response.llm_output type: {type(response.llm_output)}")
        # # print(f"{log_prefix} [DEBUG on_llm_end] response.llm_output content:")
        # try:
        #     from rich import print as rprint
    
        # except ImportError: # print(response.llm_output)

        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, "LLMEnd") # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_llm_end for {run_id}.")
                return
            outputs = {"response": response, "kwargs": kwargs}
            # --- Token Usage Extraction and Accumulation ---
            token_usage = None
            prompt_tokens = None  # Use standard name
            completion_tokens = None # Use standard name
            total_tokens = None
            try:
                if response.llm_output and isinstance(response.llm_output, dict):
                    # Check for OpenAI/standard 'token_usage' first
                    if 'token_usage' in response.llm_output:
                        token_usage = response.llm_output.get('token_usage')
                        if token_usage and isinstance(token_usage, dict):
                            self._log(f"  Extracted OpenAI token usage for run_id={run_id}: {token_usage}")
                            prompt_tokens = token_usage.get('prompt_tokens')
                            completion_tokens = token_usage.get('completion_tokens')
                            total_tokens = token_usage.get('total_tokens') # OpenAI provides total
                    # Check for Anthropic 'usage'
                    elif 'usage' in response.llm_output:
                        token_usage = response.llm_output.get('usage')
                        if token_usage and isinstance(token_usage, dict):
                            self._log(f"  Extracted Anthropic token usage for run_id={run_id}: {token_usage}")
                            prompt_tokens = token_usage.get('input_tokens') # Anthropic uses input_tokens
                            completion_tokens = token_usage.get('output_tokens') # Anthropic uses output_tokens
                            # Calculate total if possible
                            if prompt_tokens is not None and completion_tokens is not None:
                                total_tokens = prompt_tokens + completion_tokens
                            else:
                                self._log(f"  Could not calculate total_tokens from Anthropic usage: input={prompt_tokens}, output={completion_tokens}")

                    # --- Store individual usage in span output and Accumulate --- 
                    if prompt_tokens is not None or completion_tokens is not None:
                        # Store individual usage for this span
                        outputs['usage'] = { 
                            'prompt_tokens': prompt_tokens,
                            'completion_tokens': completion_tokens,
                            'total_tokens': total_tokens
                        }
                        # Accumulate tokens for the entire trace
                        # if isinstance(prompt_tokens, int):
                        #     self._current_prompt_tokens += prompt_tokens
                        # if isinstance(completion_tokens, int):
                        #     self._current_completion_tokens += completion_tokens
                        # self._log(f"  Accumulated tokens for run_id={run_id}. Current totals: Prompt={self._current_prompt_tokens}, Completion={self._current_completion_tokens}")
                    else:
                         self._log(f"  Could not extract token usage structure from llm_output for run_id={run_id}")
                else: self._log(f"  llm_output not available/dict for run_id={run_id}")
            except Exception as e:
                self._log(f"  ERROR extracting/accumulating token usage for run_id={run_id}: {e}")
            # --- End Token Usage ---
            self._end_span_tracking(trace_client, run_id, span_type="llm", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # print(f"{log_prefix} ENTERING on_llm_error: run_id={run_id}. Parent: {parent_run_id}")

        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, "LLMError") # Corrected call
            if not trace_client:
                # print(f"{log_prefix} No trace client obtained in on_llm_error for {run_id}.")
                return
            self._end_span_tracking(trace_client, run_id, span_type="llm", error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, invocation_params: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs: Any) -> Any:
        # Reuse on_llm_start logic, adding message formatting if needed
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        chat_model_name = name or serialized.get("name", "ChatModel Call")
        # Add OPENAI_API_CALL suffix if model is OpenAI and not present
        is_openai = any(key.startswith('openai') for key in serialized.get('secrets', {}).keys()) or 'openai' in chat_model_name.lower()
        is_anthropic = any(key.startswith('anthropic') for key in serialized.get('secrets', {}).keys()) or 'anthropic' in chat_model_name.lower() or 'claude' in chat_model_name.lower()
        is_together = any(key.startswith('together') for key in serialized.get('secrets', {}).keys()) or 'together' in chat_model_name.lower()
        # Add more checks for other providers like Google if needed
        is_google = any(key.startswith('google') for key in serialized.get('secrets', {}).keys()) or 'google' in chat_model_name.lower() or 'gemini' in chat_model_name.lower()

        if is_openai and "OPENAI_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} OPENAI_API_CALL"
        elif is_anthropic and "ANTHROPIC_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} ANTHROPIC_API_CALL"
        elif is_together and "TOGETHER_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} TOGETHER_API_CALL"

        elif is_google and "GOOGLE_API_CALL" not in chat_model_name:
             chat_model_name = f"{chat_model_name} GOOGLE_API_CALL"

        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_chat_model_start: name='{chat_model_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # The call below was missing parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, chat_model_name) # Corrected call with parent_run_id
            if not trace_client: return
            inputs = {'messages': messages, 'invocation_params': invocation_params or kwargs, 'options': options, 'tags': tags, 'metadata': metadata, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, chat_model_name, span_type="llm", inputs=messages) # Use 'llm' span_type for consistency
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chat_model_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    # --- Agent Methods (Async versions - ensure parent_run_id passed if needed) ---
    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_agent_action: tool={action.tool}, run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            # Optional: Implement detailed tracing if needed
            pass
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_agent_action for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_agent_finish: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            # Optional: Implement detailed tracing if needed
            pass
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_agent_finish for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

# --- Async Handler --- 

# --- NEW Fully Functional Async Handler ---
class AsyncJudgevalCallbackHandler(AsyncCallbackHandler):
    """
    Async LangChain Callback Handler using run_id/parent_run_id for hierarchy.
    Manages its own internal TraceClient instance created upon first use.
    Includes verbose logging and defensive checks.
    """
    lc_serializable = False
    lc_kwargs = {}

    def __init__(self, tracer: Tracer):
        instance_id = id(self)
        # print(f"{HANDLER_LOG_PREFIX} *** Async Handler instance {instance_id} __init__ called. ***")
        self.tracer = tracer
        self._trace_client: Optional[TraceClient] = None
        self._run_id_to_span_id: Dict[UUID, str] = {}
        self._span_id_to_start_time: Dict[str, float] = {}
        self._span_id_to_depth: Dict[str, int] = {}
        self._run_id_to_context_token: Dict[UUID, contextvars.Token] = {} # Initialize missing attribute
        self._root_run_id: Optional[UUID] = None
        self._trace_context_token: Optional[contextvars.Token] = None # Restore missing attribute
        self._trace_saved: bool = False # <<< ADDED MISSING ATTRIBUTE
        self._run_id_to_start_inputs: Dict[UUID, Dict] = {} # <<< ADDED input storage

        # --- Token Count Accumulators ---
        # self._current_prompt_tokens = 0
        # self._current_completion_tokens = 0
        # --- End Token Count Accumulators ---

        self.executed_nodes: List[str] = []
        self.executed_tools: List[str] = []
        self.executed_node_tools: List[str] = []
        self.traces: List[Dict[str, Any]] = []

    # NOTE: _ensure_trace_client remains synchronous as it doesn't involve async I/O
    def _ensure_trace_client(self, run_id: UUID, event_name: str) -> Optional[TraceClient]:
        """Ensures the internal trace client is initialized. Returns client or None."""
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        if self._trace_client is None:
            trace_id = str(uuid.uuid4())
            project = self.tracer.project_name
            try:
                client_instance = TraceClient(
                    self.tracer, trace_id, event_name, project_name=project,
                    overwrite=False, rules=self.tracer.rules,
                    enable_monitoring=self.tracer.enable_monitoring,
                    enable_evaluations=self.tracer.enable_evaluations
                )
                self._trace_client = client_instance
                if self._trace_client:
                     self.tracer._active_trace_client = self._trace_client
                     self._current_trace_id = self._trace_client.trace_id
                     if self._root_run_id is None:
                         self._root_run_id = run_id
                else:
                    return None
            except Exception as e:
                self._trace_client = None
                return None
        return self._trace_client

    def _log(self, message: str):
        """Helper for consistent logging format."""
        pass

    # NOTE: _start_span_tracking remains mostly synchronous, TraceClient.add_entry might become async later
    def _start_span_tracking(
        self,
        trace_client: TraceClient,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        span_type: SpanType = "span",
        inputs: Optional[Dict[str, Any]] = None
    ):
        self._log(f"_start_span_tracking called for: name='{name}', run_id={run_id}, parent_run_id={parent_run_id}, span_type={span_type}")
        if not trace_client:
            handler_instance_id = id(self)
            log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
            self._log(f"{log_prefix} FATAL ERROR in _start_span_tracking: trace_client argument is None for name='{name}', run_id={run_id}. Aborting span start.")
            return

        # --- NEW: Set trace context variable if not already set for this trace --- 
        if self._trace_context_token is None:
            try:
                self._trace_context_token = current_trace_var.set(trace_client)
                self._log(f"  Set current_trace_var for trace_id {trace_client.trace_id}")
            except Exception as e:
                self._log(f"  ERROR setting current_trace_var for trace_id {trace_client.trace_id}: {e}")
        # --- END NEW --- 

        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        trace_client_instance_id = id(trace_client) if trace_client else 'None'
        # print(f"{log_prefix} _start_span_tracking: Using TraceClient ID: {trace_client_instance_id}")

        start_time = time.time()
        span_id = str(uuid.uuid4())
        parent_span_id: Optional[str] = None
        current_depth = 0

        if parent_run_id and parent_run_id in self._run_id_to_span_id:
            parent_span_id = self._run_id_to_span_id[parent_run_id]
            if parent_span_id in self._span_id_to_depth:
                current_depth = self._span_id_to_depth[parent_span_id] + 1
            else:
                self._log(f"  WARNING: Parent span depth not found for parent_span_id: {parent_span_id}. Setting depth to 0.")
        elif parent_run_id:
            self._log(f"  WARNING: parent_run_id {parent_run_id} provided for '{name}' ({run_id}) but parent span not tracked. Treating as depth 0.")
        else:
            self._log(f"  No parent_run_id provided. Treating '{name}' as depth 0.")

        self._run_id_to_span_id[run_id] = span_id
        self._span_id_to_start_time[span_id] = start_time
        self._span_id_to_depth[span_id] = current_depth
        self._log(f"  Tracking new span: span_id={span_id}, depth={current_depth}")

        # --- Set SPAN context variable ONLY for chain (node) spans ---
        if span_type == "chain":
            try:
                # Set current_span_var for the node's execution context
                token = current_span_var.set(span_id) # Store the token
                self._run_id_to_context_token[run_id] = token # Store token in the dictionary
                self._log(f"  Set current_span_var to {span_id} for run_id {run_id} (type: chain)")
            except Exception as e:
                self._log(f"  ERROR setting current_span_var for run_id {run_id}: {e}")
        # --- END Span Context Var Logic ---

        try:
            # TODO: Check if trace_client.add_entry needs await if TraceClient becomes async
            trace_client.add_entry(TraceEntry(
                type="enter", span_id=span_id, trace_id=trace_client.trace_id,
                parent_span_id=parent_span_id, function=name, depth=current_depth,
                message=name, created_at=start_time, span_type=span_type
            ))
            self._log(f"  Added 'enter' entry for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'enter' entry for span_id {span_id}: {e}")
            # print(traceback.format_exc())

        if inputs:
            # _record_input_data is also sync for now
            self._record_input_data(trace_client, run_id, inputs)

    # NOTE: _end_span_tracking remains mostly synchronous, TraceClient.save might become async later
    def _end_span_tracking(
        self,
        trace_client: TraceClient,
        run_id: UUID,
        span_type: SpanType = "span",
        outputs: Optional[Any] = None,
        error: Optional[BaseException] = None
    ):
        # self._log(f"_end_span_tracking called for: run_id={run_id}, span_type={span_type}")

        # --- Define instance_id early for logging/cleanup ---
        instance_id = id(self)

        if not trace_client:
            # Use instance_id defined above
            # log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {instance_id}]"
            # self._log(f"{log_prefix} FATAL ERROR in _end_span_tracking: trace_client argument is None for run_id={run_id}. Aborting span end.")
            return

        # Use instance_id defined above
        # log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {instance_id}]"
        # trace_client_instance_id = id(trace_client) if trace_client else 'None'
        # # print(f"{log_prefix} _end_span_tracking: Using TraceClient ID: {trace_client_instance_id}")

        if run_id not in self._run_id_to_span_id:
            # self._log(f"  WARNING: Attempting to end span for untracked run_id: {run_id}")
            # Allow root run end to proceed for cleanup/save attempt even if span wasn't tracked
            if run_id != self._root_run_id:
                 return
            else:
                 # self._log(f"  Allowing root run {run_id} end logic to proceed despite untracked span.")
                 span_id = None # Indicate span wasn't found for duration/metadata lookup
        else:
            span_id = self._run_id_to_span_id[run_id]

        start_time = self._span_id_to_start_time.get(span_id) if span_id else None
        depth = self._span_id_to_depth.get(span_id, 0) if span_id else 0 # Use 0 depth if span_id is None
        duration = time.time() - start_time if start_time is not None else None
        # self._log(f"  Ending span for run_id={run_id} (span_id={span_id}). Start time={start_time}, Duration={duration}, Depth={depth}")

        # Record output/error first
        if error:
            # self._log(f"  Recording error for run_id={run_id} (span_id={span_id}): {error}")
            self._record_output_data(trace_client, run_id, error)
        elif outputs is not None:
            # output_repr = repr(outputs)
            # log_output = (output_repr[:100] + '...') if len(output_repr) > 103 else output_repr
            # self._log(f"  Recording output for run_id={run_id} (span_id={span_id}): {log_output}")
            self._record_output_data(trace_client, run_id, outputs)

        # Add exit entry (only if span was tracked)
        if span_id:
            entry_function_name = "unknown"
            try:
                if hasattr(trace_client, 'entries') and trace_client.entries:
                    entry_function_name = next((e.function for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), "unknown")
                else:
                    # self._log(f"  WARNING: Cannot determine function name for exit span_id {span_id}, trace_client.entries missing or empty.")
                    pass
            except Exception as e:
                # self._log(f"  ERROR finding function name for exit entry span_id {span_id}: {e}")
                # # print(traceback.format_exc())
                pass

            try:
                trace_client.add_entry(TraceEntry(
                    type="exit", span_id=span_id, trace_id=trace_client.trace_id,
                    depth=depth, created_at=time.time(), duration=duration,
                    span_type=span_type, function=entry_function_name
                ))
                # self._log(f"  Added 'exit' entry for span_id={span_id}, function='{entry_function_name}'")
            except Exception as e:
                # self._log(f"  ERROR adding 'exit' entry for span_id {span_id}: {e}")
                # # print(traceback.format_exc())
                pass

            # Clean up dictionaries for this specific span
            if span_id in self._span_id_to_start_time: del self._span_id_to_start_time[span_id]
            if span_id in self._span_id_to_depth: del self._span_id_to_depth[span_id]

            # Pop context token (Sync version) but don't reset
            token = self._run_id_to_context_token.pop(run_id, None)
            if token:
                # self._log(f" Popped token for run_id {run_id} (was {span_id}), not resetting context var.")
                pass
        else:
             # self._log(f"  Skipping exit entry and cleanup for run_id {run_id} as span_id was not found.")
             pass

        # Check if this is the root run ending
        if run_id == self._root_run_id:
            trace_saved_successfully = False # Track save success
            try:
                # Reset root run id after attempt
                self._root_run_id = None
                # Reset input storage for this handler instance
                self._run_id_to_start_inputs = {}
                self._log(f"Reset root run ID and input storage for handler {instance_id}.")

                self._log(f"Root run {run_id} finished. Attempting to save trace...")
                if self._trace_client and not self._trace_saved: # Check if not already saved
                    try:
                        # TODO: Check if trace_client.save needs await if TraceClient becomes async
                        trace_id, trace_data = self._trace_client.save(overwrite=self._trace_client.overwrite) # Use client's overwrite setting
                        self.traces.append(trace_data)
                        self._log(f"Trace {trace_id} successfully saved.")
                        self._trace_saved = True # Set flag only after successful save
                        trace_saved_successfully = True # Mark success
                    except Exception as e:
                        self._log(f"ERROR saving trace {self._trace_client.trace_id}: {e}")
                        # print(traceback.format_exc())
                    # REMOVED FINALLY BLOCK THAT RESET STATE HERE
                elif self._trace_client and self._trace_saved:
                     self._log(f"  Trace {self._trace_client.trace_id} already saved. Skipping save.")
                else:
                    self._log(f"  WARNING: Root run {run_id} ended, but trace client was None. Cannot save trace.")
            finally:
                # --- NEW: Consolidated Cleanup Logic --- 
                # This block executes regardless of save success/failure
                self._log(f"  Performing cleanup for root run {run_id} in handler {instance_id}.")
                # Reset root run id
                self._root_run_id = None
                # Reset input storage for this handler instance
                self._run_id_to_start_inputs = {}
                # Reset tracer's active client ONLY IF it was this handler's client
                if self.tracer._active_trace_client == self._trace_client:
                    self.tracer._active_trace_client = None
                    self._log("  Reset active_trace_client on Tracer.")
                # Completely remove trace_context_token cleanup as it's not used in sync handler
                # Optionally: Reset the entire trace client instance for this handler?
                # self._trace_client = None # Uncomment if handler should reset client completely after root run
                self._log(f"  Cleanup complete for root run {run_id}.")
                # --- End Cleanup Logic ---

    # NOTE: _record_input_data remains synchronous for now
    def _record_input_data(self,
                           trace_client: TraceClient,
                           run_id: UUID,
                           inputs: Dict[str, Any]):
        # self._log(f"_record_input_data called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            # self._log(f"  WARNING: Attempting to record input for untracked run_id: {run_id}")
            return
        if not trace_client:
             # self._log(f"  ERROR: TraceClient is None when trying to record input for run_id={run_id}")
             return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        # self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            # Find the corresponding 'enter' entry to get the function name and span type
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                # self._log(f"  Found function='{function_name}', span_type='{span_type}' for input span_id={span_id}")
            else:
                # self._log(f"  WARNING: Could not find 'enter' entry for input span_id={span_id}")
                pass
        except Exception as e:
            # self._log(f"  ERROR finding enter entry for input span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

        try:
            input_entry = TraceEntry(
                type="input",
                span_id=span_id,
                trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None), # Get parent from enter entry
                function=function_name,
                depth=depth,
                message=f"Input to {function_name}",
                created_at=time.time(),
                inputs=inputs,
                span_type=span_type
            )
            trace_client.add_entry(input_entry)
            # self._log(f"  Added 'input' entry directly for span_id={span_id}")
        except Exception as e:
            # self._log(f"  ERROR adding 'input' entry directly for span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

    # NOTE: _record_output_data remains synchronous for now
    def _record_output_data(self,
                            trace_client: TraceClient,
                            run_id: UUID,
                            output: Any):
        self._log(f"_record_output_data called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            # self._log(f"  WARNING: Attempting to record output for untracked run_id: {run_id}")
            return
        if not trace_client:
            # self._log(f"  ERROR: TraceClient is None when trying to record output for run_id={run_id}")
            return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        # self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            # Find the corresponding 'enter' entry to get the function name and span type
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                # self._log(f"  Found function='{function_name}', span_type='{span_type}' for output span_id={span_id}")
            else:
                 # self._log(f"  WARNING: Could not find 'enter' entry for output span_id={span_id}")
                 pass
        except Exception as e:
            # self._log(f"  ERROR finding enter entry for output span_id {span_id}: {e}")
            # # print(traceback.format_exc())
            pass

        try:
            output_entry = TraceEntry(
                type="output",
                span_id=span_id,
                trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None), # Get parent from enter entry
                function=function_name,
                depth=depth,
                message=f"Output from {function_name}",
                created_at=time.time(),
                output=output, # Langchain outputs are typically serializable directly
                span_type=span_type
            )
            trace_client.add_entry(output_entry)
            self._log(f"  Added 'output' entry directly for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'output' entry directly for span_id {span_id}: {e}")
            # print(traceback.format_exc())

    # --- Async Callback Methods ---

    async def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        serialized_name = serialized.get('name', 'Unknown') if serialized else "Unknown (Serialized=None)"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_retriever_start: name='{serialized_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            name = f"RETRIEVER_{(serialized_name).upper()}"
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, name) # Corrected call
            if not trace_client: return
            inputs = {'query': query, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type="retriever", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_retriever_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_retriever_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, parent_run_id, "RetrieverEnd") # Corrected call
            if not trace_client: return
            doc_summary = [{"index": i, "page_content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(documents)]
            outputs = {"document_count": len(documents), "documents": doc_summary, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="retriever", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_retriever_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        # Handle potential None for serialized safely
        serialized_name = serialized.get('name') if serialized else None
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # Log the potentially generic or specific name found in serialized
        log_name = serialized_name if serialized_name else "Unknown (Serialized=None)"
        # print(f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}] ENTERING on_chain_start: serialized_name='{log_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            # Determine the best name and span type
            name = "Unknown Chain" # Default
            span_type: SpanType = "chain"
            node_name = metadata.get("langgraph_node") if metadata else None
            is_langgraph_root = kwargs.get('name') == 'LangGraph' # Check kwargs
            is_potential_root_event = parent_run_id is None

            # Define generic names to ignore if node_name is not present
            GENERIC_NAMES = ["RunnableSequence", "RunnableParallel", "RunnableLambda", "LangGraph", "__start__", "__end__"]

            if node_name:
                name = node_name
                self._log(f"  LangGraph Node Start Detected: '{name}', run_id={run_id}, parent_run_id={parent_run_id}")
                if name not in self.executed_nodes: self.executed_nodes.append(name)
            elif serialized_name and serialized_name not in GENERIC_NAMES:
                name = serialized_name
                self._log(f"  LangGraph Functional Step (Router?): '{name}', run_id={run_id}, parent_run_id={parent_run_id}")
            # Correct root detection: Should primarily rely on parent_run_id being None for the *first* event.
            # kwargs name='LangGraph' might appear later.
            elif is_potential_root_event: # Check if it's the potential root event
                 # Use the serialized name if available and not generic, otherwise default to 'LangGraph'
                 if serialized_name and serialized_name not in GENERIC_NAMES:
                     name = serialized_name
                 else:
                     name = "LangGraph"
                 self._log(f"  LangGraph Root Start Detected (parent_run_id=None): Name='{name}', run_id={run_id}")
                 if self._root_run_id is None: # Only set root_run_id once
                    self._log(f"  Setting root run ID to {run_id}")
                    self._root_run_id = run_id
                 # Defer trace client name update until client is ensured
            elif serialized_name: # Fallback if node_name missing and serialized_name was generic or root wasn't detected
                name = serialized_name
                self._log(f"  Fallback to serialized_name: '{name}', run_id={run_id}")

            # Ensure trace client exists (using the determined name for initialization if needed)
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, name) # FIXED: Removed parent_run_id
            if not trace_client:
                 # print(f"{log_prefix} No trace client obtained in on_chain_start for {run_id} ('{name}').")
                 return

            # --- Update Trace Name if Root (Moved After Client Ensure) ---
            if is_potential_root_event and run_id == self._root_run_id and trace_client.name != name:
                self._log(f"  Updating trace name from '{trace_client.name}' to '{name}' for root run {run_id}")
                trace_client.name = name
            # --- End Update Trace Name ---

            # Start span tracking using the determined name and span_type
            combined_inputs = {'inputs': inputs, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type=span_type, inputs=combined_inputs)
            # --- Store inputs for potential evaluation later --- 
            self._run_id_to_start_inputs[run_id] = inputs # Store the raw inputs dict
            self._log(f"  Stored inputs for run_id {run_id}")
            # --- End Store inputs --- 

        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **kwargs: Any) -> Any:
        """
        Ends span tracking for a chain/node and attempts evaluation if applicable.
        """
        # --- Existing logging and client check ---
        instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {instance_id}]"
        self._log(f"{log_prefix} ENTERING on_chain_end: run_id={run_id}. Current TraceClient ID: {id(self._trace_client) if self._trace_client else 'None'}")
        client = self._ensure_trace_client(run_id, "on_chain_end") # Ensure client exists
        if not client:
            self._log(f"{log_prefix} No TraceClient found for on_chain_end ({run_id}). Aborting.")
            return # Early exit if no client

        # --- Get span_id associated with this chain run ---
        span_id = self._run_id_to_span_id.get(run_id)

        # --- Existing span ending logic --- 
        # Determine span_type for end_span_tracking (copied from sync handler)
        end_span_type: SpanType = "chain" # Default
        if span_id: # Check if span_id was actually found
            try:
                if hasattr(client, 'entries') and client.entries:
                    enter_entry = next((e for e in reversed(client.entries) if e.span_id == span_id and e.type == "enter"), None)
                    if enter_entry: end_span_type = enter_entry.span_type
                else: self._log(f"  WARNING: trace_client.entries empty/missing for on_chain_end span_id={span_id}")
            except Exception as e:
                self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_end: {e}")
        else:
             self._log(f"  WARNING: No span_id found for run_id {run_id} in on_chain_end, using default span_type='chain'.")

        # Prepare outputs for end tracking (moved down)
        combined_outputs = {"outputs": outputs, "tags": tags, "kwargs": kwargs}

        # Call end_span_tracking with potentially determined span_type
        self._end_span_tracking(client, run_id, span_type=end_span_type, outputs=combined_outputs)

        # --- Root node cleanup REMOVED - Now handled in _end_span_tracking ---

        # --- NEW: Attempt Evaluation by checking output metadata --- 
        eval_config: Optional[EvaluationConfig] = None
        node_name = "unknown_node" # Default node name
        # Ensure client exists before proceeding with eval logic that uses it
        if client: 
            if span_id: # Try to find the node name from the 'enter' entry
                 try:
                     if hasattr(client, 'entries') and client.entries:
                         enter_entry = next((e for e in reversed(client.entries) if e.span_id == span_id and e.type == "enter"), None)
                         if enter_entry: node_name = enter_entry.function
                 except Exception as e:
                     self._log(f"  ERROR finding node name for span_id {span_id} in on_chain_end: {e}")
            
            if span_id and "_judgeval_eval" in outputs: # Only attempt if span exists and key is present
                raw_eval_config = outputs.get("_judgeval_eval")
                if isinstance(raw_eval_config, EvaluationConfig):
                    eval_config = raw_eval_config
                    self._log(f"{log_prefix} Found valid EvaluationConfig in outputs for node='{node_name}'.")
                elif isinstance(raw_eval_config, dict):
                     # Attempt to reconstruct from dict if needed (e.g., if state serialization occurred)
                     try:
                         # Basic check for required keys before attempting reconstruction
                         if "scorers" in raw_eval_config and "example" in raw_eval_config:
                             # Example might also be a dict, try reconstructing it
                             example_data = raw_eval_config["example"]
                             reconstructed_example = Example(**example_data) if isinstance(example_data, dict) else example_data

                             if isinstance(reconstructed_example, Example):
                                 eval_config = EvaluationConfig(
                                     scorers=raw_eval_config["scorers"], # Assumes scorers are serializable or passed correctly
                                     example=reconstructed_example,
                                     model=raw_eval_config.get("model"),
                                     log_results=raw_eval_config.get("log_results", True)
                                 )
                                 self._log(f"{log_prefix} Reconstructed EvaluationConfig from dict in outputs for node='{node_name}'.")
                             else:
                                self._log(f"{log_prefix} Could not reconstruct Example from dict in _judgeval_eval for node='{node_name}'. Skipping evaluation.")
                         else:
                             self._log(f"{log_prefix} Dict in _judgeval_eval missing required keys ('scorers', 'example') for node='{node_name}'. Skipping evaluation.")
                     except Exception as recon_e:
                         self._log(f"{log_prefix} ERROR attempting to reconstruct EvaluationConfig from dict for node='{node_name}': {recon_e}")
                         # print(traceback.format_exc()) # Print traceback for reconstruction errors
                else:
                    self._log(f"{log_prefix} Found '_judgeval_eval' key in outputs for node='{node_name}', but it wasn't an EvaluationConfig object or reconstructable dict. Skipping evaluation.")


            if eval_config and span_id: # Check eval_config *and* span_id again
                self._log(f"{log_prefix} Submitting evaluation for span_id={span_id}")
                try:
                    # Ensure example has trace_id set if not already present
                    if not hasattr(eval_config.example, 'trace_id') or not eval_config.example.trace_id:
                        # Use the correct variable name 'client' here for the async handler
                        eval_config.example.trace_id = client.trace_id 
                        self._log(f"{log_prefix} Set trace_id={client.trace_id} on evaluation example.")
                    
                    # Call async_evaluate on the TraceClient instance ('client')
                    # Use the correct variable name 'client' here for the async handler
                    client.async_evaluate( 
                        scorers=eval_config.scorers,
                        example=eval_config.example,
                        model=eval_config.model,
                        log_results=eval_config.log_results,
                        span_id=span_id # Pass the specific span_id for this node run
                    )
                    self._log(f"{log_prefix} Evaluation submitted successfully for span_id={span_id}.")
                except Exception as eval_e:
                    self._log(f"{log_prefix} ERROR submitting evaluation for span_id={span_id}: {eval_e}")
                    # print(traceback.format_exc()) # Print traceback for evaluation errors
            elif "_judgeval_eval" in outputs and not span_id:
                 self._log(f"{log_prefix} WARNING: Found _judgeval_eval in outputs, but span_id for run_id {run_id} was not found. Cannot submit evaluation.")
        # --- End NEW Evaluation Logic ---

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_chain_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, "ChainError") # FIXED: Removed parent_run_id
            if not trace_client: return

            span_id = self._run_id_to_span_id.get(run_id)
            span_type: SpanType = "chain"
            if span_id:
                try:
                     if hasattr(trace_client, 'entries') and trace_client.entries:
                        enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                        if enter_entry: span_type = enter_entry.span_type
                     else: self._log(f"  WARNING: trace_client.entries not available for on_chain_error span_id={span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_error: {e}")

            self._end_span_tracking(trace_client, run_id, span_type=span_type, error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        # Handle potential None for serialized
        name = serialized.get("name", "Unnamed Tool") if serialized else "Unknown Tool (Serialized=None)"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_tool_start: name='{name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, name) # FIXED: Removed parent_run_id
            if not trace_client: return

            combined_inputs = {'input_str': input_str, 'inputs': inputs, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type="tool", inputs=combined_inputs)

            # --- Track executed tools (logic remains the same) ---
            if name not in self.executed_tools: self.executed_tools.append(name)
            parent_node_name = None
            if parent_run_id and parent_run_id in self._run_id_to_span_id:
                parent_span_id = self._run_id_to_span_id[parent_run_id]
                try:
                    if hasattr(trace_client, 'entries') and trace_client.entries:
                        parent_enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == parent_span_id and e.type == "enter" and e.span_type == "chain"), None)
                        if parent_enter_entry:
                            parent_node_name = parent_enter_entry.function
                    else:
                        self._log(f"  WARNING: trace_client.entries not available for parent node {parent_span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding parent node name for tool start span_id {parent_span_id}: {e}")

            node_tool = f"{parent_node_name}:{name}" if parent_node_name else name
            if node_tool not in self.executed_node_tools: self.executed_node_tools.append(node_tool)
            self._log(f"  Tracked node_tool: {node_tool}")
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_tool_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, "ToolEnd") # FIXED: Removed parent_run_id
            if not trace_client: return
            outputs = {"output": output, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="tool", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_tool_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, "ToolError") # FIXED: Removed parent_run_id
            if not trace_client: return
            self._end_span_tracking(trace_client, run_id, span_type="tool", error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, invocation_params: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        llm_name = name or serialized.get("name", "LLM Call")
        # print(f"{log_prefix} ENTERING on_llm_start: name='{llm_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, llm_name) # FIXED: Removed parent_run_id
            if not trace_client: return
            inputs = {'prompts': prompts, 'invocation_params': invocation_params or kwargs, 'options': options, 'tags': tags, 'metadata': metadata, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, llm_name, span_type="llm", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'

        try:
            trace_client = self._ensure_trace_client(run_id, "LLMEnd")
            if not trace_client:
                return

            outputs = {"response": response, "kwargs": kwargs}
            # --- Token Usage Extraction and Accumulation ---
            token_usage = None
            prompt_tokens = None  # Use standard name
            completion_tokens = None # Use standard name
            total_tokens = None
            try:
                if response.llm_output and isinstance(response.llm_output, dict):
                    # Check for OpenAI/standard 'token_usage' first
                    if 'token_usage' in response.llm_output:
                        token_usage = response.llm_output.get('token_usage')
                        if token_usage and isinstance(token_usage, dict):
                            self._log(f"  Extracted OpenAI token usage for run_id={run_id}: {token_usage}")
                            prompt_tokens = token_usage.get('prompt_tokens')
                            completion_tokens = token_usage.get('completion_tokens')
                            total_tokens = token_usage.get('total_tokens')
                    # Check for Anthropic 'usage'
                    elif 'usage' in response.llm_output:
                        token_usage = response.llm_output.get('usage')
                        if token_usage and isinstance(token_usage, dict):
                            self._log(f"  Extracted Anthropic token usage for run_id={run_id}: {token_usage}")
                            prompt_tokens = token_usage.get('input_tokens') # Anthropic uses input_tokens
                            completion_tokens = token_usage.get('output_tokens') # Anthropic uses output_tokens
                            # Calculate total if possible
                            if prompt_tokens is not None and completion_tokens is not None:
                                total_tokens = prompt_tokens + completion_tokens
                            else:
                                self._log(f"  Could not calculate total_tokens from Anthropic usage: input={prompt_tokens}, output={completion_tokens}")

                    # Add to outputs if any tokens were found
                    if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
                        outputs['usage'] = { # Add under 'usage' key
                            'prompt_tokens': prompt_tokens, # Use standard keys
                            'completion_tokens': completion_tokens,
                            'total_tokens': total_tokens
                        }
                    else:
                         self._log(f"  Could not extract token usage structure from llm_output for run_id={run_id}")
                else: self._log(f"  llm_output not available/dict for run_id={run_id}")
            except Exception as e:
                self._log(f"  ERROR extracting token usage for run_id={run_id}: {e}")

            self._end_span_tracking(trace_client, run_id, span_type="llm", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_llm_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Pass parent_run_id
            trace_client = self._ensure_trace_client(run_id, "LLMError") # FIXED: Removed parent_run_id
            if not trace_client: return
            self._end_span_tracking(trace_client, run_id, span_type="llm", error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, invocation_params: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs: Any) -> Any:
        # Reuse on_llm_start logic, adding message formatting if needed
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        chat_model_name = name or serialized.get("name", "ChatModel Call")
        # Add OPENAI_API_CALL suffix if model is OpenAI and not present
        is_openai = any(key.startswith('openai') for key in serialized.get('secrets', {}).keys()) or 'openai' in chat_model_name.lower()
        is_anthropic = any(key.startswith('anthropic') for key in serialized.get('secrets', {}).keys()) or 'anthropic' in chat_model_name.lower() or 'claude' in chat_model_name.lower()
        is_together = any(key.startswith('together') for key in serialized.get('secrets', {}).keys()) or 'together' in chat_model_name.lower()
        # Add more checks for other providers like Google if needed
        is_google = any(key.startswith('google') for key in serialized.get('secrets', {}).keys()) or 'google' in chat_model_name.lower() or 'gemini' in chat_model_name.lower()

        if is_openai and "OPENAI_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} OPENAI_API_CALL"
        elif is_anthropic and "ANTHROPIC_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} ANTHROPIC_API_CALL"
        elif is_together and "TOGETHER_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} TOGETHER_API_CALL"
        # Add elif for Google: check for 'google' or 'gemini'?
        # elif is_google and "GOOGLE_API_CALL" not in chat_model_name:
        #     chat_model_name = f"{chat_model_name} GOOGLE_API_CALL"
        elif is_google and "GOOGLE_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} GOOGLE_API_CALL"

        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_chat_model_start: name='{chat_model_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # trace_client = self._ensure_trace_client(run_id, parent_run_id, chat_model_name) # Corrected call << INCORRECT COMMENT
            trace_client = self._ensure_trace_client(run_id, chat_model_name) # FIXED: Removed parent_run_id
            if not trace_client: return
            inputs = {'messages': messages, 'invocation_params': invocation_params or kwargs, 'options': options, 'tags': tags, 'metadata': metadata, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, chat_model_name, span_type="llm", inputs=inputs) # Use 'llm' span_type for consistency
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chat_model_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            # print(traceback.format_exc())

    # --- Agent Methods (Async versions - ensure parent_run_id passed if needed) ---
    async def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_agent_action: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # trace_client = self._ensure_trace_client(run_id, parent_run_id, "AgentAction") # Corrected call << INCORRECT COMMENT
            trace_client = self._ensure_trace_client(run_id, "AgentAction") # FIXED: Removed parent_run_id
            if not trace_client: return
            # inputs = {\"action\": action, \"kwargs\": kwargs}
            inputs = {"action": action, "kwargs": kwargs} # FIXED: Removed bad escapes
            # Agent actions often lead to tool calls, treat as a distinct step
            # self._start_span_tracking(trace_client, run_id, parent_run_id, name=\"AgentAction\", span_type=\"agent_action\", inputs=inputs)
            self._start_span_tracking(trace_client, run_id, parent_run_id, name="AgentAction", span_type="agent_action", inputs=inputs) # FIXED: Removed bad escapes
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            # self._log(f\"{log_prefix} UNCAUGHT EXCEPTION in on_agent_action for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}\")
            self._log(f'{log_prefix} UNCAUGHT EXCEPTION in on_agent_action for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}') # FIXED: Changed f-string quotes
            # print(traceback.format_exc())


    async def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # print(f"{log_prefix} ENTERING on_agent_finish: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # trace_client = self._ensure_trace_client(run_id, parent_run_id, "AgentFinish") # Corrected call << INCORRECT COMMENT
            trace_client = self._ensure_trace_client(run_id, "AgentFinish") # FIXED: Removed parent_run_id
            if not trace_client: return
            # outputs = {\"finish\": finish, \"kwargs\": kwargs}
            outputs = {"finish": finish, "kwargs": kwargs} # FIXED: Removed bad escapes
            # Corresponds to the end of an AgentAction span? Or a chain span? Assuming agent_action here.
            # self._end_span_tracking(trace_client, run_id, span_type=\"agent_action\", outputs=outputs)
            self._end_span_tracking(trace_client, run_id, span_type="agent_action", outputs=outputs) # FIXED: Removed bad escapes
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            # self._log(f\"{log_prefix} UNCAUGHT EXCEPTION in on_agent_finish for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}\")
            self._log(f'{log_prefix} UNCAUGHT EXCEPTION in on_agent_finish for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}') # FIXED: Changed f-string quotes
            # print(traceback.format_exc())