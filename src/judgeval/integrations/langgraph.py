from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID
import time
import uuid
from datetime import datetime, timezone

from judgeval.common.tracer import (
    TraceClient,
    TraceSpan,
    Tracer,
    SpanType,
    cost_per_token,
)
from judgeval.data.trace import TraceUsage

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_core.messages.base import BaseMessage
from langchain_core.documents import Document

# --- Get context vars from tracer module ---
# Assuming tracer.py defines these and they are accessible
# If not, redefine them here or adjust import

# from judgeval.common.tracer import current_span_var
# TODO: Figure out how to handle context variables. Current solution is to keep track of current span id in Tracer class


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
    lc_kwargs: dict = {}

    # --- NEW __init__ ---
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        # Initialize tracking/logging variables (preserved across resets)
        self.executed_nodes: List[str] = []
        self.executed_tools: List[str] = []
        self.executed_node_tools: List[str] = []
        self.traces: List[Dict[str, Any]] = []
        # Initialize execution state (reset between runs)
        self._reset_state()

    # --- END NEW __init__ ---

    def _reset_state(self):
        """Reset only the critical execution state for reuse across multiple executions"""
        # Reset core execution state that must be cleared between runs
        self._trace_client: Optional[TraceClient] = None
        self._run_id_to_span_id: Dict[UUID, str] = {}
        self._span_id_to_start_time: Dict[str, float] = {}
        self._span_id_to_depth: Dict[str, int] = {}
        self._root_run_id: Optional[UUID] = None
        self._trace_saved: bool = False  # Flag to prevent actions after trace is saved
        self.span_id_to_token: Dict[str, Any] = {}
        self.trace_id_to_token: Dict[str, Any] = {}

        # Add timestamp to track when we last reset
        self._last_reset_time: float = time.time()

        # Preserve tracking/logging variables across executions:
        # - self.executed_nodes: List[str] = [] # Keep as running log
        # - self.executed_tools: List[str] = [] # Keep as running log
        # - self.executed_node_tools: List[str] = [] # Keep as running log
        # - self.traces: List[Dict[str, Any]] = [] # Keep for collecting multiple traces

        # Also reset tracking/logging variables
        self.executed_nodes: List[
            str
        ] = []  # These last four members are only appended to and never accessed; can probably be removed but still might be useful for future reference?
        self.executed_tools: List[str] = []
        self.executed_node_tools: List[str] = []
        self.traces: List[Dict[str, Any]] = []

    # --- END NEW __init__ ---
    def reset(self):
        """Public method to manually reset handler execution state for reuse"""
        self._reset_state()

    def reset_all(self):
        """Public method to reset ALL handler state including tracking/logging data"""
        self._reset_state()

    # --- MODIFIED _ensure_trace_client ---
    def _ensure_trace_client(
        self, run_id: UUID, parent_run_id: Optional[UUID], event_name: str
    ) -> Optional[TraceClient]:
        """
        Ensures the internal trace client is initialized, creating it only once
        per handler instance lifecycle (effectively per graph invocation).
        Returns the client or None.
        """

        # If this is a potential new root execution (no parent_run_id) and we had a previous trace saved,
        # reset state to allow reuse of the handler
        if parent_run_id is None and self._trace_saved:
            self._reset_state()

        # If a client already exists, return it.
        if self._trace_client:
            return self._trace_client

        # If no client exists, initialize it NOW.
        trace_id = str(uuid.uuid4())
        project = self.tracer.project_name
        try:
            # Use event_name as the initial trace name, might be updated later by on_chain_start if root
            client_instance = TraceClient(
                self.tracer,
                trace_id,
                event_name,
                project_name=project,
                enable_monitoring=self.tracer.enable_monitoring,
                enable_evaluations=self.tracer.enable_evaluations,
            )
            self._trace_client = client_instance
            token = self.tracer.set_current_trace(self._trace_client)
            if token:
                self.trace_id_to_token[trace_id] = token
            if self._trace_client:
                self._root_run_id = (
                    run_id  # Assign the first run_id encountered as the tentative root
                )
                self._trace_saved = False  # Ensure flag is reset
                # Set active client on Tracer (important for potential fallbacks)
                self.tracer._active_trace_client = self._trace_client

                # NEW: Initial save for live tracking (follows the new practice)
                try:
                    trace_id_saved, server_response = self._trace_client.save(
                        final_save=False,  # Initial save for live tracking
                    )
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to save initial trace for live tracking: {e}"
                    )

                return self._trace_client
            else:
                return None
        except Exception:
            self._trace_client = None
            self._root_run_id = None
            return None

    def _start_span_tracking(
        self,
        trace_client: TraceClient,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        span_type: SpanType = "span",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start tracking a span, ensuring trace client exists"""

        start_time = time.time()
        span_id = str(uuid.uuid4())
        parent_span_id: Optional[str] = None
        current_depth = 0

        if parent_run_id and parent_run_id in self._run_id_to_span_id:
            parent_span_id = self._run_id_to_span_id[parent_run_id]
            if parent_span_id in self._span_id_to_depth:
                current_depth = self._span_id_to_depth[parent_span_id] + 1

        self._run_id_to_span_id[run_id] = span_id
        self._span_id_to_start_time[span_id] = start_time
        self._span_id_to_depth[span_id] = current_depth

        new_span = TraceSpan(
            span_id=span_id,
            trace_id=trace_client.trace_id,
            parent_span_id=parent_span_id,
            function=name,
            depth=current_depth,
            created_at=start_time,
            span_type=span_type,
        )

        # Separate metadata from inputs
        if inputs:
            metadata = {}
            clean_inputs = {}

            # Extract metadata fields
            metadata_fields = ["tags", "metadata", "kwargs", "serialized"]
            for field in metadata_fields:
                if field in inputs:
                    metadata[field] = inputs.pop(field)

            # Store the remaining inputs
            clean_inputs = inputs

            # Set both fields on the span
            new_span.inputs = clean_inputs
            new_span.additional_metadata = metadata
            new_span.increment_update_id()  # Thread-safe increment for span modification
        else:
            new_span.inputs = {}
            new_span.additional_metadata = {}

        trace_client.add_span(new_span)

        # Queue span with initial state (input phase) through background service
        if trace_client.background_span_service:
            trace_client.background_span_service.queue_span(
                new_span, span_state="input"
            )

        token = self.tracer.set_current_span(span_id)
        if token:
            self.span_id_to_token[span_id] = token

    def _end_span_tracking(
        self,
        trace_client: TraceClient,
        run_id: UUID,
        outputs: Optional[Any] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """End tracking a span, ensuring trace client exists"""

        # Get span ID and check if it exists
        span_id = self._run_id_to_span_id.get(run_id)
        if span_id:
            token = self.span_id_to_token.pop(span_id, None)
            self.tracer.reset_current_span(token, span_id)

        start_time = self._span_id_to_start_time.get(span_id) if span_id else None
        duration = time.time() - start_time if start_time is not None else None

        # Add exit entry (only if span was tracked)
        if span_id:
            trace_span = trace_client.span_id_to_span.get(span_id)
            if trace_span:
                trace_span.duration = duration
                trace_span.increment_update_id()  # Thread-safe increment for span modification

                # Handle outputs and error
                if error:
                    trace_span.output = error
                    trace_span.increment_update_id()  # Thread-safe increment for span modification
                elif outputs:
                    # Separate metadata from outputs
                    metadata = {}
                    clean_outputs = {}

                    # Extract metadata fields
                    metadata_fields = ["tags", "kwargs"]
                    if isinstance(outputs, dict):
                        for field in metadata_fields:
                            if field in outputs:
                                metadata[field] = outputs.pop(field)

                        # Store the remaining outputs
                        clean_outputs = outputs
                    else:
                        clean_outputs = outputs

                    # Set both fields on the span
                    trace_span.output = clean_outputs
                    trace_span.increment_update_id()  # Thread-safe increment for span modification
                    if metadata:
                        # Merge with existing metadata
                        existing_metadata = trace_span.additional_metadata or {}
                        trace_span.additional_metadata = {
                            **existing_metadata,
                            **metadata,
                        }
                        trace_span.increment_update_id()  # Thread-safe increment for span modification

                # Queue span with completed state through background service
                if trace_client.background_span_service:
                    span_state = "error" if error else "completed"
                    trace_client.background_span_service.queue_span(
                        trace_span, span_state=span_state
                    )

            # Clean up dictionaries for this specific span
            if span_id in self._span_id_to_start_time:
                del self._span_id_to_start_time[span_id]
            if span_id in self._span_id_to_depth:
                del self._span_id_to_depth[span_id]

        # Check if this is the root run ending
        if run_id == self._root_run_id:
            try:
                # Reset root run id after attempt
                self._root_run_id = None
                # Reset input storage for this handler instance

                if (
                    self._trace_client and not self._trace_saved
                ):  # Check if not already saved
                    # Flush background spans before saving the final trace

                    complete_trace_data = {
                        "trace_id": self._trace_client.trace_id,
                        "name": self._trace_client.name,
                        "created_at": datetime.fromtimestamp(
                            self._trace_client.start_time, timezone.utc
                        ).isoformat(),
                        "duration": self._trace_client.get_duration(),
                        "trace_spans": [
                            span.model_dump() for span in self._trace_client.trace_spans
                        ],
                        "offline_mode": self.tracer.offline_mode,
                        "parent_trace_id": self._trace_client.parent_trace_id,
                        "parent_name": self._trace_client.parent_name,
                    }
                    trace_id, trace_data = self._trace_client.save(
                        final_save=True,  # Final save with usage counter updates
                    )
                    token = self.trace_id_to_token.pop(trace_id, None)
                    self.tracer.reset_current_trace(token, trace_id)

                    # Store complete trace data instead of server response
                    self.tracer.traces.append(complete_trace_data)
                    self._trace_saved = True  # Set flag only after successful save
            finally:
                # --- NEW: Consolidated Cleanup Logic ---
                # This block executes regardless of save success/failure
                # Reset root run id
                self._root_run_id = None
                # Reset input storage for this handler instance
                if self.tracer._active_trace_client == self._trace_client:
                    self.tracer._active_trace_client = None
                # --- End Cleanup Logic ---

    # --- Callback Methods ---
    # Each method now ensures the trace client exists before proceeding

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        serialized_name = (
            serialized.get("name", "Unknown")
            if serialized
            else "Unknown (Serialized=None)"
        )

        name = f"RETRIEVER_{(serialized_name).upper()}"
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, name
        )  # Corrected call
        if not trace_client:
            return

        inputs = {
            "query": query,
            "tags": tags,
            "metadata": metadata,
            "kwargs": kwargs,
            "serialized": serialized,
        }
        self._start_span_tracking(
            trace_client,
            run_id,
            parent_run_id,
            name,
            span_type="retriever",
            inputs=inputs,
        )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, "RetrieverEnd"
        )  # Corrected call
        if not trace_client:
            return
        doc_summary = [
            {
                "index": i,
                "page_content": doc.page_content[:100] + "..."
                if len(doc.page_content) > 100
                else doc.page_content,
                "metadata": doc.metadata,
            }
            for i, doc in enumerate(documents)
        ]
        outputs = {
            "document_count": len(documents),
            "documents": doc_summary,
            "kwargs": kwargs,
        }
        self._end_span_tracking(trace_client, run_id, outputs=outputs)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        serialized_name = (
            serialized.get("name") if serialized else "Unknown (Serialized=None)"
        )

        # --- Determine Name and Span Type ---
        span_type: SpanType = "chain"
        name = serialized_name if serialized_name else "Unknown Chain"  # Default name
        node_name = metadata.get("langgraph_node") if metadata else None
        is_langgraph_root_kwarg = (
            kwargs.get("name") == "LangGraph"
        )  # Check kwargs for explicit root name
        # More robust root detection: Often the first chain event with parent_run_id=None *is* the root.
        is_potential_root_event = parent_run_id is None

        if node_name:
            name = node_name  # Use node name if available
            if name not in self.executed_nodes:
                self.executed_nodes.append(
                    name
                )  # Leaving this in for now but can probably be removed
        elif is_langgraph_root_kwarg and is_potential_root_event:
            name = "LangGraph"  # Explicit root detected
        # Add handling for other potential LangChain internal chains if needed, e.g., "RunnableSequence"

        # --- Ensure Trace Client ---
        # Pass parent_run_id to _ensure_trace_client
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, name
        )  # Corrected call
        if not trace_client:
            return

        # --- Update Trace Name if Root ---
        # If this is the root event (parent_run_id is None) and the trace client was just created,
        # ensure the trace name reflects the graph's name ('LangGraph' usually).
        if (
            is_potential_root_event
            and run_id == self._root_run_id
            and trace_client.name != name
        ):
            trace_client.name = name  # Update trace name to the determined root name

        # --- Start Span Tracking ---
        combined_inputs = {
            "inputs": inputs,
            "tags": tags,
            "metadata": metadata,
            "kwargs": kwargs,
            "serialized": serialized,
        }
        self._start_span_tracking(
            trace_client,
            run_id,
            parent_run_id,
            name,
            span_type=span_type,
            inputs=combined_inputs,
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, "ChainEnd"
        )  # Corrected call
        if not trace_client:
            return

        span_id = self._run_id_to_span_id.get(run_id)
        # If it's the root run ending, _end_span_tracking will handle cleanup/save
        if not span_id and run_id != self._root_run_id:
            return  # Don't call end tracking if it's not the root and span wasn't tracked

        # Prepare outputs for end tracking (moved down)
        combined_outputs = {"outputs": outputs, "tags": tags, "kwargs": kwargs}

        # Call end_span_tracking with potentially determined span_type
        self._end_span_tracking(trace_client, run_id, outputs=combined_outputs)

        # --- Root node cleanup (Existing logic - slightly modified save call) ---
        if run_id == self._root_run_id:
            if trace_client and not self._trace_saved:
                # Store complete trace data instead of server response
                complete_trace_data = {
                    "trace_id": trace_client.trace_id,
                    "name": trace_client.name,
                    "created_at": datetime.fromtimestamp(
                        trace_client.start_time, timezone.utc
                    ).isoformat(),
                    "duration": trace_client.get_duration(),
                    "trace_spans": [
                        span.model_dump() for span in trace_client.trace_spans
                    ],
                    "offline_mode": self.tracer.offline_mode,
                    "parent_trace_id": trace_client.parent_trace_id,
                    "parent_name": trace_client.parent_name,
                }
                trace_id_saved, trace_data = trace_client.save(
                    final_save=True,
                )

                self.tracer.traces.append(complete_trace_data)
                self._trace_saved = True
                # Reset tracer's active client *after* successful save
                if self.tracer._active_trace_client == trace_client:
                    self.tracer._active_trace_client = None

            # Reset root run id after attempt
            self._root_run_id = None
            # Reset input storage for this handler instance

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, "ChainError"
        )  # Corrected call
        if not trace_client:
            return

        span_id = self._run_id_to_span_id.get(run_id)

        # Let _end_span_tracking handle potential root run cleanup
        if not span_id and run_id != self._root_run_id:
            return

        self._end_span_tracking(trace_client, run_id, error=error)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        name = (
            serialized.get("name", "Unnamed Tool")
            if serialized
            else "Unknown Tool (Serialized=None)"
        )

        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, name
        )  # Corrected call
        if not trace_client:
            return

        combined_inputs = {
            "input_str": input_str,
            "inputs": inputs,
            "tags": tags,
            "metadata": metadata,
            "kwargs": kwargs,
            "serialized": serialized,
        }
        self._start_span_tracking(
            trace_client,
            run_id,
            parent_run_id,
            name,
            span_type="tool",
            inputs=combined_inputs,
        )

        # --- Track executed tools (remains the same) ---
        if name not in self.executed_tools:
            self.executed_tools.append(
                name
            )  # Leaving this in for now but can probably be removed
        parent_node_name = None
        if parent_run_id and parent_run_id in self._run_id_to_span_id:
            parent_span_id = self._run_id_to_span_id[parent_run_id]
            parent_node_name = trace_client.span_id_to_span[parent_span_id].function

        node_tool = f"{parent_node_name}:{name}" if parent_node_name else name
        if node_tool not in self.executed_node_tools:
            self.executed_node_tools.append(
                node_tool
            )  # Leaving this in for now but can probably be removed
        # --- End Track executed tools ---

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, "ToolEnd"
        )  # Corrected call
        if not trace_client:
            return
        outputs = {"output": output, "kwargs": kwargs}
        self._end_span_tracking(trace_client, run_id, outputs=outputs)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, "ToolError"
        )  # Corrected call
        if not trace_client:
            return
        self._end_span_tracking(trace_client, run_id, error=error)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        llm_name = name or serialized.get("name", "LLM Call")

        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, llm_name
        )  # Corrected call
        if not trace_client:
            return
        inputs = {
            "prompts": prompts,
            "invocation_params": invocation_params or kwargs,
            "options": options,
            "tags": tags,
            "metadata": metadata,
            "serialized": serialized,
        }
        self._start_span_tracking(
            trace_client,
            run_id,
            parent_run_id,
            llm_name,
            span_type="llm",
            inputs=inputs,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, "LLMEnd"
        )  # Corrected call
        if not trace_client:
            return
        outputs = {"response": response, "kwargs": kwargs}

        # --- Token Usage Extraction and Cost Calculation ---
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        model_name = None

        # Extract model name from response if available
        if (
            hasattr(response, "llm_output")
            and response.llm_output
            and isinstance(response.llm_output, dict)
        ):
            model_name = response.llm_output.get(
                "model_name"
            ) or response.llm_output.get("model")

        # Try to get model from the first generation if available
        if not model_name and response.generations and len(response.generations) > 0:
            if (
                hasattr(response.generations[0][0], "generation_info")
                and response.generations[0][0].generation_info
            ):
                gen_info = response.generations[0][0].generation_info
                model_name = gen_info.get("model") or gen_info.get("model_name")

        if response.llm_output and isinstance(response.llm_output, dict):
            # Check for OpenAI/standard 'token_usage' first
            if "token_usage" in response.llm_output:
                token_usage = response.llm_output.get("token_usage")
                if token_usage and isinstance(token_usage, dict):
                    prompt_tokens = token_usage.get("prompt_tokens")
                    completion_tokens = token_usage.get("completion_tokens")
                    total_tokens = token_usage.get(
                        "total_tokens"
                    )  # OpenAI provides total
            # Check for Anthropic 'usage'
            elif "usage" in response.llm_output:
                token_usage = response.llm_output.get("usage")
                if token_usage and isinstance(token_usage, dict):
                    prompt_tokens = token_usage.get(
                        "input_tokens"
                    )  # Anthropic uses input_tokens
                    completion_tokens = token_usage.get(
                        "output_tokens"
                    )  # Anthropic uses output_tokens
                    # Calculate total if possible
                    if prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens

            # --- Create TraceUsage object and set on span ---
            if prompt_tokens is not None or completion_tokens is not None:
                # Calculate costs if model name is available
                prompt_cost = None
                completion_cost = None
                total_cost_usd = None

                if (
                    model_name
                    and prompt_tokens is not None
                    and completion_tokens is not None
                ):
                    try:
                        prompt_cost, completion_cost = cost_per_token(
                            model=model_name,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                        total_cost_usd = (
                            (prompt_cost + completion_cost)
                            if prompt_cost and completion_cost
                            else None
                        )
                    except Exception as e:
                        # If cost calculation fails, continue without costs
                        import warnings

                        warnings.warn(
                            f"Failed to calculate token costs for model {model_name}: {e}"
                        )

                # Create TraceUsage object
                usage = TraceUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                    or (
                        prompt_tokens + completion_tokens
                        if prompt_tokens and completion_tokens
                        else None
                    ),
                    prompt_tokens_cost_usd=prompt_cost,
                    completion_tokens_cost_usd=completion_cost,
                    total_cost_usd=total_cost_usd,
                    model_name=model_name,
                )

                # Set usage on the actual span (not in outputs)
                span_id = self._run_id_to_span_id.get(run_id)
                if span_id and span_id in trace_client.span_id_to_span:
                    trace_span = trace_client.span_id_to_span[span_id]
                    trace_span.usage = usage
                    trace_span.increment_update_id()  # Thread-safe increment for span modification

        self._end_span_tracking(trace_client, run_id, outputs=outputs)
        # --- End Token Usage ---

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, "LLMError"
        )  # Corrected call
        if not trace_client:
            return
        self._end_span_tracking(trace_client, run_id, error=error)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        # Reuse on_llm_start logic, adding message formatting if needed
        chat_model_name = name or serialized.get("name", "ChatModel Call")
        # Add OPENAI_API_CALL suffix if model is OpenAI and not present
        is_openai = (
            any(
                key.startswith("openai") for key in serialized.get("secrets", {}).keys()
            )
            or "openai" in chat_model_name.lower()
        )
        is_anthropic = (
            any(
                key.startswith("anthropic")
                for key in serialized.get("secrets", {}).keys()
            )
            or "anthropic" in chat_model_name.lower()
            or "claude" in chat_model_name.lower()
        )
        is_together = (
            any(
                key.startswith("together")
                for key in serialized.get("secrets", {}).keys()
            )
            or "together" in chat_model_name.lower()
        )
        # Add more checks for other providers like Google if needed
        is_google = (
            any(
                key.startswith("google") for key in serialized.get("secrets", {}).keys()
            )
            or "google" in chat_model_name.lower()
            or "gemini" in chat_model_name.lower()
        )

        if is_openai and "OPENAI_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} OPENAI_API_CALL"
        elif is_anthropic and "ANTHROPIC_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} ANTHROPIC_API_CALL"
        elif is_together and "TOGETHER_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} TOGETHER_API_CALL"

        elif is_google and "GOOGLE_API_CALL" not in chat_model_name:
            chat_model_name = f"{chat_model_name} GOOGLE_API_CALL"

        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, chat_model_name
        )  # Corrected call with parent_run_id
        if not trace_client:
            return
        inputs = {
            "messages": messages,
            "invocation_params": invocation_params or kwargs,
            "options": options,
            "tags": tags,
            "metadata": metadata,
            "serialized": serialized,
        }
        self._start_span_tracking(
            trace_client,
            run_id,
            parent_run_id,
            chat_model_name,
            span_type="llm",
            inputs=inputs,
        )  # Use 'llm' span_type for consistency

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        action_tool = action.tool
        name = f"AGENT_ACTION_{(action_tool).upper()}"
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, name
        )  # Corrected call
        if not trace_client:
            return

        inputs = {
            "tool_input": action.tool_input,
            "log": action.log,
            "messages": action.messages,
            "kwargs": kwargs,
        }
        self._start_span_tracking(
            trace_client, run_id, parent_run_id, name, span_type="agent", inputs=inputs
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Pass parent_run_id
        trace_client = self._ensure_trace_client(
            run_id, parent_run_id, "AgentFinish"
        )  # Corrected call
        if not trace_client:
            return

        outputs = {
            "return_values": finish.return_values,
            "log": finish.log,
            "messages": finish.messages,
            "kwargs": kwargs,
        }
        self._end_span_tracking(trace_client, run_id, outputs=outputs)
