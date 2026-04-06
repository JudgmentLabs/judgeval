from abc import ABC


try:
    import openlit  # type: ignore
except ImportError:
    raise ImportError(
        "Openlit is not installed and required for the openlit integration. Please install it with `pip install openlit`."
    )


class Openlit(ABC):
    """Integration that routes OpenLit instrumentation through Judgment.

    OpenLit provides auto-instrumentation for many LLM providers and
    vector databases. This class connects OpenLit's tracer to the active
    Judgment tracer provider so all spans appear in your Judgment dashboard.

    Examples:
        ```python
        from judgeval import Tracer
        from judgeval.integrations import Openlit

        Tracer.init(project_name="my-agent")
        Openlit.initialize()
        ```
    """

    @staticmethod
    def initialize(**kwargs):
        """Activate OpenLit instrumentation through Judgment.

        Args:
            **kwargs: Additional arguments forwarded to ``openlit.init()``.
                Metrics collection is disabled by default since Judgment
                only consumes trace data.
        """
        from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
        from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

        openlit.init(
            tracer=JudgmentTracerProvider.get_instance().get_tracer(
                JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
            ),
            disable_metrics=True,
            **kwargs,
        )


__all__ = ["Openlit"]
