from abc import ABC
from judgeval.v1.trace import Tracer
from judgeval.logger import judgeval_logger
from judgeval.utils.url import url_for


try:
    import openlit  # type: ignore
except ImportError:
    raise ImportError(
        "Openlit is not installed and required for the openlit integration. Please install it with `pip install openlit`."
    )


class Openlit(ABC):
    @staticmethod
    def initialize(
        tracer: Tracer,
        **kwargs,
    ):
        api_key = tracer.api_key
        organization_id = tracer.organization_id
        project_name = tracer.project_name

        if not project_name:
            judgeval_logger.warning(
                "Project name not provided. Openlit will not be initialized."
            )
            return

        if not tracer._client:
            judgeval_logger.warning(
                "Tracer client not configured. Openlit will not be initialized."
            )
            return

        project_id = tracer.project_id
        if not project_id:
            judgeval_logger.warning(
                f"Project {project_name} failed to resolve. Openlit will not be initialized."
            )
            return

        from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME

        openlit.init(
            service_name=project_name,
            otlp_endpoint=url_for("/otel"),
            otlp_headers={
                "Authorization": f"Bearer {api_key}",
                "X-Organization-Id": organization_id,
                "X-Project-Id": project_id,
            },
            otel_tracer=tracer._tracer_provider.get_tracer(
                JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
            ),
            disable_metrics=True,
            **kwargs,
        )


__all__ = ["Openlit"]
