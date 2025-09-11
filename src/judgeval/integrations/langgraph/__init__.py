from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set, Type
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs import LLMResult, ChatGeneration
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError(
        "Judgeval's langgraph integration requires langchain to be installed. Please install it with `pip install judgeval[langchain]`"
    ) from e

from judgeval.tracer import Tracer
from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.managers import sync_span_context
from judgeval.utils.serialize import safe_serialize
from judgeval.logger import judgeval_logger
from opentelemetry.trace import Status, StatusCode, Span

# Control flow exception types that should not be treated as errors
CONTROL_FLOW_EXCEPTION_TYPES: Set[Type[BaseException]] = set()

try:
    from langgraph.errors import GraphBubbleUp

    CONTROL_FLOW_EXCEPTION_TYPES.add(GraphBubbleUp)
except ImportError:
    pass

LANGSMITH_TAG_HIDDEN: str = "langsmith:hidden"


class JudgevalCallbackHandler(BaseCallbackHandler):
    """
    LangGraph/LangChain Callback Handler that creates OpenTelemetry spans
    using the Judgeval tracer framework.

    This handler tracks the execution of chains, tools, LLMs, and other components
    in a LangGraph/LangChain application, creating proper span hierarchies for monitoring.
    """

    # Prevent LangChain serialization issues
    lc_serializable = False
    lc_kwargs: dict = {}

    def __init__(self, tracer: Optional[Tracer] = None):
        """
        Initialize the callback handler.

        Args:
            tracer: Optional Tracer instance. If not provided, will try to use an active tracer.
        """
        self.tracer = tracer
        if self.tracer is None:
            # Try to get an active tracer
            if Tracer._active_tracers:
                self.tracer = next(iter(Tracer._active_tracers))
            else:
                judgeval_logger.warning(
                    "No tracer provided and no active tracers found. "
                    "Callback handler will not create spans."
                )
                return

        # Track spans by run_id for proper hierarchy
        self.spans: Dict[UUID, Span] = {}
        self.span_start_times: Dict[UUID, float] = {}
        self.run_id_to_span_id: Dict[UUID, str] = {}
        self.span_id_to_depth: Dict[str, int] = {}
        self.root_run_id: Optional[UUID] = None

        # Track execution for debugging
        self.executed_nodes: List[str] = []
        self.executed_tools: List[str] = []
        self.executed_node_tools: List[Dict[str, Any]] = []

    def reset(self):
        """Reset handler state for reuse across multiple executions."""
        self.spans.clear()
        self.span_start_times.clear()
        self.executed_nodes.clear()
        self.executed_tools.clear()
        self.executed_node_tools.clear()

    def _get_run_name(self, serialized: Optional[Dict[str, Any]], **kwargs: Any) -> str:
        """Extract the name of the operation from serialized data or kwargs."""
        if "name" in kwargs and kwargs["name"] is not None:
            return str(kwargs["name"])

        if serialized is None:
            return "<unknown>"

        try:
            return str(serialized["name"])
        except (KeyError, TypeError):
            pass

        try:
            return str(serialized["id"][-1])
        except (KeyError, TypeError):
            pass

        return "<unknown>"

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert a LangChain message to a dictionary for storage."""
        if isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, ToolMessage):
            message_dict = {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }
        elif isinstance(message, FunctionMessage):
            message_dict = {"role": "function", "content": message.content}
        elif isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        else:
            message_dict = {"role": "unknown", "content": str(message.content)}

        if hasattr(message, "additional_kwargs") and message.additional_kwargs:
            message_dict["additional_kwargs"] = str(message.additional_kwargs)

        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Convert a list of LangChain messages to dictionaries."""
        return [self._convert_message_to_dict(m) for m in messages]

    def _join_tags_and_metadata(
        self,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Join tags and metadata into a single dictionary."""
        final_dict = {}
        if tags is not None and len(tags) > 0:
            final_dict["tags"] = tags
        if metadata is not None:
            final_dict.update(metadata)
        return final_dict if final_dict else None

    def _start_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        span_type: str,
        inputs: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra_attributes: Any,
    ) -> None:
        """Start a new span for the given run."""
        if not self.tracer:
            return

        # Skip internal spans
        if name.startswith("__") and name.endswith("__"):
            return

        try:
            # Determine if this is a root span
            is_root = parent_run_id is None
            if is_root:
                self.root_run_id = run_id

            # Calculate depth for proper hierarchy
            current_depth = 0
            if parent_run_id and parent_run_id in self.run_id_to_span_id:
                parent_span_id = self.run_id_to_span_id[parent_run_id]
                current_depth = self.span_id_to_depth.get(parent_span_id, 0) + 1

            # Create span attributes
            attributes = {
                AttributeKeys.JUDGMENT_SPAN_KIND.value: span_type,
            }

            # Add metadata and tags
            combined_metadata = self._join_tags_and_metadata(tags, metadata)
            if combined_metadata:
                metadata_str = safe_serialize(combined_metadata)
                attributes["metadata"] = metadata_str

            # Add extra attributes
            for key, value in extra_attributes.items():
                if value is not None:
                    attributes[str(key)] = str(value)

            # Create span using the tracer's context manager for proper hierarchy
            with sync_span_context(self.tracer, name, attributes) as span:
                # Set input data if provided
                if inputs is not None:
                    span.set_attribute(
                        AttributeKeys.JUDGMENT_INPUT.value, safe_serialize(inputs)
                    )

                # Store span information for tracking
                span_id = (
                    str(span.get_span_context().span_id)
                    if span.get_span_context()
                    else str(uuid.uuid4())
                )
                self.spans[run_id] = span
                self.span_start_times[run_id] = time.time()
                self.run_id_to_span_id[run_id] = span_id
                self.span_id_to_depth[span_id] = current_depth

        except Exception as e:
            judgeval_logger.exception(f"Error starting span for {name}: {e}")

    def _end_span(
        self,
        run_id: UUID,
        outputs: Any = None,
        error: Optional[BaseException] = None,
        **extra_attributes: Any,
    ) -> None:
        """End the span for the given run."""
        if run_id not in self.spans:
            return

        try:
            span = self.spans[run_id]

            # Set output data if provided
            if outputs is not None:
                span.set_attribute(
                    AttributeKeys.JUDGMENT_OUTPUT.value, safe_serialize(outputs)
                )

            # Set additional attributes
            for key, value in extra_attributes.items():
                if value is not None:
                    span.set_attribute(str(key), str(value))

            # Handle errors
            if error is not None:
                # Check if this is a control flow exception
                is_control_flow = any(
                    isinstance(error, t) for t in CONTROL_FLOW_EXCEPTION_TYPES
                )
                if not is_control_flow:
                    span.record_exception(error)
                    span.set_status(Status(StatusCode.ERROR, str(error)))
                # Control flow exceptions don't set error status
            else:
                span.set_status(Status(StatusCode.OK))

            # Note: The span will be ended automatically by the context manager

        except Exception as e:
            judgeval_logger.exception(f"Error ending span for run_id {run_id}: {e}")
        finally:
            # Cleanup tracking data
            if run_id in self.spans:
                del self.spans[run_id]
            if run_id in self.span_start_times:
                del self.span_start_times[run_id]
            if run_id in self.run_id_to_span_id:
                span_id = self.run_id_to_span_id[run_id]
                del self.run_id_to_span_id[run_id]
                if span_id in self.span_id_to_depth:
                    del self.span_id_to_depth[span_id]

            # Check if this is the root run ending
            if run_id == self.root_run_id:
                self.root_run_id = None

    def _log_debug_event(
        self,
        event_name: str,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log debug information about callback events."""
        judgeval_logger.debug(
            f"Event: {event_name}, run_id: {str(run_id)[:8]}, "
            f"parent_run_id: {str(parent_run_id)[:8] if parent_run_id else None}"
        )

    # Chain callbacks
    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain starts running."""
        try:
            self._log_debug_event(
                "on_chain_start", run_id, parent_run_id, inputs=inputs
            )

            name = self._get_run_name(serialized, **kwargs)

            # Check for LangGraph node
            node_name = metadata.get("langgraph_node") if metadata else None
            if node_name:
                name = node_name
                if name not in self.executed_nodes:
                    self.executed_nodes.append(name)

            # Determine if this is a root LangGraph execution
            is_langgraph_root = (
                kwargs.get("name") == "LangGraph" and parent_run_id is None
            )
            if is_langgraph_root:
                name = "LangGraph"

            span_level = "DEBUG" if tags and LANGSMITH_TAG_HIDDEN in tags else None

            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=name,
                span_type="chain",
                inputs=inputs,
                tags=tags,
                metadata=metadata,
                level=span_level,
                serialized=safe_serialize(serialized) if serialized else None,
            )
        except Exception as e:
            judgeval_logger.exception(f"Error in on_chain_start: {e}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain ends successfully."""
        try:
            self._log_debug_event(
                "on_chain_end", run_id, parent_run_id, outputs=outputs
            )
            self._end_span(run_id=run_id, outputs=outputs)
        except Exception as e:
            judgeval_logger.exception(f"Error in on_chain_end: {e}")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain encounters an error."""
        try:
            self._log_debug_event("on_chain_error", run_id, parent_run_id, error=error)
            self._end_span(run_id=run_id, error=error)
        except Exception as e:
            judgeval_logger.exception(f"Error in on_chain_error: {e}")

    # LLM callbacks
    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when an LLM starts generating."""
        try:
            self._log_debug_event(
                "on_llm_start", run_id, parent_run_id, prompts=prompts
            )

            name = self._get_run_name(serialized, **kwargs)
            model_name = self._extract_model_name(serialized, kwargs)

            prompt_data = prompts[0] if len(prompts) == 1 else prompts

            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=name,
                span_type="llm",
                inputs=prompt_data,
                tags=tags,
                metadata=metadata,
                model=model_name,
                serialized=safe_serialize(serialized) if serialized else None,
            )

            # Set GenAI specific attributes
            if run_id in self.spans:
                span = self.spans[run_id]
                if model_name:
                    span.set_attribute(AttributeKeys.GEN_AI_REQUEST_MODEL, model_name)
                span.set_attribute(
                    AttributeKeys.GEN_AI_PROMPT, safe_serialize(prompt_data)
                )

                # Set model parameters if available
                invocation_params = kwargs.get("invocation_params", {})
                if "temperature" in invocation_params:
                    span.set_attribute(
                        AttributeKeys.GEN_AI_REQUEST_TEMPERATURE,
                        float(invocation_params["temperature"]),
                    )
                if "max_tokens" in invocation_params:
                    span.set_attribute(
                        AttributeKeys.GEN_AI_REQUEST_MAX_TOKENS,
                        int(invocation_params["max_tokens"]),
                    )

        except Exception as e:
            judgeval_logger.exception(f"Error in on_llm_start: {e}")

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chat model starts generating."""
        try:
            self._log_debug_event(
                "on_chat_model_start", run_id, parent_run_id, messages=messages
            )

            name = self._get_run_name(serialized, **kwargs)
            model_name = self._extract_model_name(serialized, kwargs)

            # Flatten messages
            flattened_messages = []
            for message_list in messages:
                flattened_messages.extend(self._create_message_dicts(message_list))

            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=name,
                span_type="llm",
                inputs=flattened_messages,
                tags=tags,
                metadata=metadata,
                model=model_name,
                serialized=safe_serialize(serialized) if serialized else None,
            )

            # Set GenAI specific attributes
            if run_id in self.spans:
                span = self.spans[run_id]
                if model_name:
                    span.set_attribute(AttributeKeys.GEN_AI_REQUEST_MODEL, model_name)
                span.set_attribute(
                    AttributeKeys.GEN_AI_PROMPT, safe_serialize(flattened_messages)
                )

        except Exception as e:
            judgeval_logger.exception(f"Error in on_chat_model_start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when an LLM finishes generating."""
        try:
            self._log_debug_event(
                "on_llm_end", run_id, parent_run_id, response=response
            )

            # Extract response content
            output: Any
            if response.generations:
                last_generation = response.generations[-1][-1]
                if (
                    isinstance(last_generation, ChatGeneration)
                    and last_generation.message
                ):
                    output = self._convert_message_to_dict(last_generation.message)
                else:
                    output = (
                        last_generation.text
                        if hasattr(last_generation, "text")
                        else str(last_generation)
                    )
            else:
                output = ""

            # Extract usage information
            usage_attrs = {}
            if response.llm_output and "token_usage" in response.llm_output:
                token_usage = response.llm_output["token_usage"]
                if hasattr(token_usage, "prompt_tokens"):
                    usage_attrs[AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS] = (
                        token_usage.prompt_tokens
                    )
                if hasattr(token_usage, "completion_tokens"):
                    usage_attrs[AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS] = (
                        token_usage.completion_tokens
                    )

            # Set completion attribute
            if run_id in self.spans:
                span = self.spans[run_id]
                span.set_attribute(
                    AttributeKeys.GEN_AI_COMPLETION, safe_serialize(output)
                )

                # Set usage attributes
                for key, value in usage_attrs.items():
                    span.set_attribute(key, value)

            self._end_span(run_id=run_id, outputs=output, **usage_attrs)  # type: ignore

        except Exception as e:
            judgeval_logger.exception(f"Error in on_llm_end: {e}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when an LLM encounters an error."""
        try:
            self._log_debug_event("on_llm_error", run_id, parent_run_id, error=error)
            self._end_span(run_id=run_id, error=error)
        except Exception as e:
            judgeval_logger.exception(f"Error in on_llm_error: {e}")

    # Tool callbacks
    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a tool starts executing."""
        try:
            self._log_debug_event(
                "on_tool_start", run_id, parent_run_id, input_str=input_str
            )

            name = self._get_run_name(serialized, **kwargs)
            if name not in self.executed_tools:
                self.executed_tools.append(name)

            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=name,
                span_type="tool",
                inputs=input_str,
                tags=tags,
                metadata=metadata,
                serialized=safe_serialize(serialized) if serialized else None,
            )
        except Exception as e:
            judgeval_logger.exception(f"Error in on_tool_start: {e}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a tool finishes executing."""
        try:
            self._log_debug_event("on_tool_end", run_id, parent_run_id, output=output)
            self._end_span(run_id=run_id, outputs=output)
        except Exception as e:
            judgeval_logger.exception(f"Error in on_tool_end: {e}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a tool encounters an error."""
        try:
            self._log_debug_event("on_tool_error", run_id, parent_run_id, error=error)
            self._end_span(run_id=run_id, error=error)
        except Exception as e:
            judgeval_logger.exception(f"Error in on_tool_error: {e}")

    # Agent callbacks
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when an agent takes an action."""
        try:
            self._log_debug_event(
                "on_agent_action", run_id, parent_run_id, action=action
            )

            if run_id in self.spans:
                span = self.spans[run_id]
                span.set_attribute("agent.action.tool", action.tool)
                span.set_attribute(
                    "agent.action.tool_input", safe_serialize(action.tool_input)
                )
                span.set_attribute("agent.action.log", action.log)

            self._end_span(
                run_id=run_id,
                outputs={"action": action.tool, "input": action.tool_input},
            )
        except Exception as e:
            judgeval_logger.exception(f"Error in on_agent_action: {e}")

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when an agent finishes."""
        try:
            self._log_debug_event(
                "on_agent_finish", run_id, parent_run_id, finish=finish
            )

            if run_id in self.spans:
                span = self.spans[run_id]
                span.set_attribute("agent.finish.log", finish.log)

            self._end_span(run_id=run_id, outputs=finish.return_values)
        except Exception as e:
            judgeval_logger.exception(f"Error in on_agent_finish: {e}")

    # Retriever callbacks
    def on_retriever_start(
        self,
        serialized: Optional[Dict[str, Any]],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a retriever starts."""
        try:
            self._log_debug_event(
                "on_retriever_start", run_id, parent_run_id, query=query
            )

            name = self._get_run_name(serialized, **kwargs)

            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=name,
                span_type="retriever",
                inputs=query,
                tags=tags,
                metadata=metadata,
                serialized=safe_serialize(serialized) if serialized else None,
            )
        except Exception as e:
            judgeval_logger.exception(f"Error in on_retriever_start: {e}")

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a retriever finishes."""
        try:
            self._log_debug_event(
                "on_retriever_end", run_id, parent_run_id, documents=documents
            )

            # Convert documents to serializable format
            doc_data = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in documents
            ]

            if run_id in self.spans:
                span = self.spans[run_id]
                span.set_attribute("retriever.document_count", len(documents))

            self._end_span(
                run_id=run_id, outputs=doc_data, document_count=len(documents)
            )
        except Exception as e:
            judgeval_logger.exception(f"Error in on_retriever_end: {e}")

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a retriever encounters an error."""
        try:
            self._log_debug_event(
                "on_retriever_error", run_id, parent_run_id, error=error
            )
            self._end_span(run_id=run_id, error=error)
        except Exception as e:
            judgeval_logger.exception(f"Error in on_retriever_error: {e}")

    def _extract_model_name(
        self, serialized: Optional[Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """Extract model name from serialized data or kwargs."""
        # Try to get from invocation params
        invocation_params = kwargs.get("invocation_params", {})
        if "model_name" in invocation_params:
            return invocation_params["model_name"]
        if "model" in invocation_params:
            return invocation_params["model"]

        # Try to get from serialized data
        if serialized:
            if "model_name" in serialized:
                return serialized["model_name"]
            if "model" in serialized:
                return serialized["model"]

        return None


__all__ = ["JudgevalCallbackHandler"]
