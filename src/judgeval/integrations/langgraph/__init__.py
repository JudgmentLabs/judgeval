from __future__ import annotations

from abc import ABC
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from judgeval.tracer import Tracer

from judgeval.integrations.langgraph.callback_handler import LangGraphCallbackHandler


class Langgraph(ABC):
    @staticmethod
    def initialize(otel_only: bool = True):
        """
        Initialize LangSmith environment variables for OTel integration.
        
        Note: This is deprecated. Use get_callback_handler() instead for
        proper OpenTelemetry context propagation.
        """
        os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        if otel_only:
            os.environ["LANGSMITH_OTEL_ONLY"] = "true"

    @staticmethod
    def get_callback_handler(tracer: Tracer, verbose: bool = False) -> LangGraphCallbackHandler:
        """
        Get a LangGraph callback handler that integrates with OpenTelemetry.
        
        This callback handler ensures proper context propagation between
        LangGraph and OpenTelemetry by creating child spans for each event.
        
        Args:
            tracer: The Judgeval Tracer instance
            verbose: Whether to log verbose information
            
        Returns:
            A configured LangGraphCallbackHandler instance
            
        Example:
            ```python
            from judgeval.tracer import Tracer
            from judgeval.integrations.langgraph import Langgraph
            from langgraph.prebuilt import create_react_agent
            
            tracer = Tracer(project_name="my-project")
            handler = Langgraph.get_callback_handler(tracer)
            
            # Use with LangGraph
            agent = create_react_agent(model, tools)
            
            @tracer.observe()
            def run_agent(query):
                return agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config={"callbacks": [handler]}
                )
            ```
        """
        return LangGraphCallbackHandler(tracer=tracer, verbose=verbose)


__all__ = ["Langgraph", "LangGraphCallbackHandler"]
