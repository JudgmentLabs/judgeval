from abc import ABC


try:
    import openlit  # type: ignore
except ImportError:
    raise ImportError(
        "Openlit is not installed and required for the openlit integration. Please install it with `pip install openlit`."
    )


class Openlit(ABC):
    @staticmethod
    def initialize(
        **kwargs,
    ):
        from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
        from judgeval.v1.trace.proxy_tracer_provider import ProxyTracerProvider

        openlit.init(
            tracer=ProxyTracerProvider.get_instance().get_tracer(
                JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
            ),
            disable_metrics=True,
            **kwargs,
        )


__all__ = ["Openlit"]
