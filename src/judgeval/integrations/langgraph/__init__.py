from __future__ import annotations

from abc import ABC
import os


class Langgraph(ABC):
    """Integration that routes LangGraph spans through the Judgment pipeline.

    Enables LangGraph's built-in OpenTelemetry export and points it at the
    active Judgment tracer provider. Call ``initialize()`` once at startup.

    Examples:
        ```python
        from judgeval import Tracer
        from judgeval.integrations import Langgraph

        Tracer.init(project_name="my-agent")
        Langgraph.initialize()
        ```
    """

    @staticmethod
    def initialize(otel_only: bool = True):
        """Activate LangGraph OTEL tracing through Judgment.

        Args:
            otel_only: If True (default), disables the LangSmith backend
                so spans are only sent to Judgment.
        """
        os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        if otel_only:
            os.environ["LANGSMITH_OTEL_ONLY"] = "true"
