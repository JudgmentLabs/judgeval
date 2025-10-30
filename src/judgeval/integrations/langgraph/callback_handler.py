"""
LangGraph callback handler that integrates with OpenTelemetry tracing.

This callback handler ensures proper OpenTelemetry context propagation
for LangGraph operations by creating OTel spans for each event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from opentelemetry import context, trace
from opentelemetry.trace import Status, StatusCode

from judgeval.logger import judgeval_logger
from judgeval.tracer.keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize

if TYPE_CHECKING:
    from judgeval.tracer import Tracer


class LangGraphCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for LangGraph that creates OpenTelemetry spans.
    
    This handler ensures proper context propagation between LangGraph
    and OpenTelemetry by creating child spans for each LangGraph event
    within the current OTel context.
    
    Usage:
        tracer = Tracer(project_name="my-project")
        handler = LangGraphCallbackHandler(tracer)
        
        # Pass to LangGraph
        agent = create_react_agent(model, tools, callbacks=[handler])
    """

    def __init__(
        self,
        tracer: Tracer,
        verbose: bool = False,
    ):
        """
        Initialize the callback handler.
        
        Args:
            tracer: The Judgeval Tracer instance
            verbose: Whether to log verbose information
        """
        super().__init__()
        self.tracer = tracer
        self.verbose = verbose
        self._run_spans: Dict[str, trace.Span] = {}
        self._context_tokens: Dict[str, object] = {}

    def _start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        run_id: Optional[UUID] = None,
    ) -> trace.Span:
        """
        Start a new OTel span as a child of the current span.
        
        This ensures proper context propagation by using the current
        OpenTelemetry context.
        """
        try:
            # Get the OTel tracer from the Judgeval tracer
            otel_tracer = self.tracer.get_tracer()
            
            # Create span in the current context
            # This automatically makes it a child of the current span
            span = otel_tracer.start_span(
                name=name,
                attributes=attributes or {},
            )
            
            # Attach the span to the context and make it current
            ctx = trace.set_span_in_context(span)
            token = context.attach(ctx)
            
            # Store the span and context token for later cleanup
            if run_id:
                run_id_str = str(run_id)
                self._run_spans[run_id_str] = span
                self._context_tokens[run_id_str] = token
            
            return span
            
        except Exception as e:
            judgeval_logger.error(f"Error starting span: {e}")
            # Return a non-recording span as fallback
            return trace.INVALID_SPAN

    def _end_span(
        self,
        run_id: UUID,
        status: Optional[Status] = None,
        output: Optional[Any] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """End a span and clean up context."""
        try:
            run_id_str = str(run_id)
            span = self._run_spans.get(run_id_str)
            
            if not span or not span.is_recording():
                return
            
            # Set output if provided
            if output is not None:
                span.set_attribute(
                    AttributeKeys.JUDGMENT_OUTPUT,
                    safe_serialize(output),
                )
            
            # Record exception if provided
            if error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
            elif status:
                span.set_status(status)
            else:
                span.set_status(Status(StatusCode.OK))
            
            # End the span
            span.end()
            
            # Detach context
            token = self._context_tokens.get(run_id_str)
            if token:
                context.detach(token)
            
            # Cleanup
            self._run_spans.pop(run_id_str, None)
            self._context_tokens.pop(run_id_str, None)
            
        except Exception as e:
            judgeval_logger.error(f"Error ending span: {e}")

    # Chain callbacks
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
        """Called when a chain starts."""
        try:
            chain_name = serialized.get("name", "Chain")
            
            attributes = {
                AttributeKeys.JUDGMENT_SPAN_KIND: "chain",
                AttributeKeys.JUDGMENT_INPUT: safe_serialize(inputs),
                "langgraph.run_id": str(run_id),
                "langgraph.chain_name": chain_name,
            }
            
            if parent_run_id:
                attributes["langgraph.parent_run_id"] = str(parent_run_id)
            
            if tags:
                attributes["langgraph.tags"] = safe_serialize(tags)
            
            if metadata:
                attributes["langgraph.metadata"] = safe_serialize(metadata)
            
            self._start_span(
                name=f"langgraph.chain.{chain_name}",
                attributes=attributes,
                run_id=run_id,
            )
            
            if self.verbose:
                judgeval_logger.info(f"Chain started: {chain_name} (run_id: {run_id})")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_chain_start: {e}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a chain ends."""
        try:
            self._end_span(run_id=run_id, output=outputs)
            
            if self.verbose:
                judgeval_logger.info(f"Chain ended (run_id: {run_id})")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_chain_end: {e}")

    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        try:
            self._end_span(run_id=run_id, error=error)
            
            if self.verbose:
                judgeval_logger.info(f"Chain error (run_id: {run_id}): {error}")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_chain_error: {e}")

    # LLM callbacks
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts."""
        try:
            model_name = serialized.get("name", "LLM")
            
            attributes = {
                AttributeKeys.JUDGMENT_SPAN_KIND: "llm",
                AttributeKeys.GEN_AI_PROMPT: safe_serialize(prompts),
                AttributeKeys.GEN_AI_REQUEST_MODEL: model_name,
                "langgraph.run_id": str(run_id),
            }
            
            if parent_run_id:
                attributes["langgraph.parent_run_id"] = str(parent_run_id)
            
            if tags:
                attributes["langgraph.tags"] = safe_serialize(tags)
            
            if metadata:
                attributes["langgraph.metadata"] = safe_serialize(metadata)
            
            # Add invocation parameters if available
            invocation_params = kwargs.get("invocation_params", {})
            if invocation_params:
                if "temperature" in invocation_params:
                    attributes[AttributeKeys.GEN_AI_REQUEST_TEMPERATURE] = invocation_params["temperature"]
                if "max_tokens" in invocation_params:
                    attributes[AttributeKeys.GEN_AI_REQUEST_MAX_TOKENS] = invocation_params["max_tokens"]
            
            self._start_span(
                name=f"langgraph.llm.{model_name}",
                attributes=attributes,
                run_id=run_id,
            )
            
            if self.verbose:
                judgeval_logger.info(f"LLM started: {model_name} (run_id: {run_id})")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_llm_start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM ends."""
        try:
            run_id_str = str(run_id)
            span = self._run_spans.get(run_id_str)
            
            if span and span.is_recording():
                # Extract completion text
                generations = response.generations
                if generations and len(generations) > 0:
                    completion = [gen.text for gen in generations[0]]
                    span.set_attribute(
                        AttributeKeys.GEN_AI_COMPLETION,
                        safe_serialize(completion),
                    )
                
                # Extract token usage if available
                llm_output = response.llm_output or {}
                if "token_usage" in llm_output:
                    usage = llm_output["token_usage"]
                    if "prompt_tokens" in usage:
                        span.set_attribute(
                            AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS,
                            usage["prompt_tokens"],
                        )
                    if "completion_tokens" in usage:
                        span.set_attribute(
                            AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                            usage["completion_tokens"],
                        )
            
            self._end_span(run_id=run_id, output=response.dict())
            
            if self.verbose:
                judgeval_logger.info(f"LLM ended (run_id: {run_id})")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_llm_end: {e}")

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM errors."""
        try:
            self._end_span(run_id=run_id, error=error)
            
            if self.verbose:
                judgeval_logger.info(f"LLM error (run_id: {run_id}): {error}")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_llm_error: {e}")

    # Tool callbacks
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts."""
        try:
            tool_name = serialized.get("name", "Tool")
            
            attributes = {
                AttributeKeys.JUDGMENT_SPAN_KIND: "tool",
                AttributeKeys.JUDGMENT_INPUT: input_str,
                "langgraph.run_id": str(run_id),
                "langgraph.tool_name": tool_name,
            }
            
            if parent_run_id:
                attributes["langgraph.parent_run_id"] = str(parent_run_id)
            
            if tags:
                attributes["langgraph.tags"] = safe_serialize(tags)
            
            if metadata:
                attributes["langgraph.metadata"] = safe_serialize(metadata)
            
            self._start_span(
                name=f"langgraph.tool.{tool_name}",
                attributes=attributes,
                run_id=run_id,
            )
            
            if self.verbose:
                judgeval_logger.info(f"Tool started: {tool_name} (run_id: {run_id})")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_tool_start: {e}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool ends."""
        try:
            self._end_span(run_id=run_id, output=output)
            
            if self.verbose:
                judgeval_logger.info(f"Tool ended (run_id: {run_id})")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_tool_end: {e}")

    def on_tool_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        try:
            self._end_span(run_id=run_id, error=error)
            
            if self.verbose:
                judgeval_logger.info(f"Tool error (run_id: {run_id}): {error}")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_tool_error: {e}")

    # Agent callbacks
    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an agent takes an action."""
        try:
            run_id_str = str(run_id)
            span = self._run_spans.get(run_id_str)
            
            if span and span.is_recording():
                span.set_attribute(
                    "langgraph.agent_action",
                    safe_serialize(action),
                )
            
            if self.verbose:
                judgeval_logger.info(f"Agent action (run_id: {run_id}): {action}")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_agent_action: {e}")

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when an agent finishes."""
        try:
            run_id_str = str(run_id)
            span = self._run_spans.get(run_id_str)
            
            if span and span.is_recording():
                span.set_attribute(
                    "langgraph.agent_finish",
                    safe_serialize(finish),
                )
            
            if self.verbose:
                judgeval_logger.info(f"Agent finish (run_id: {run_id}): {finish}")
                
        except Exception as e:
            judgeval_logger.error(f"Error in on_agent_finish: {e}")
