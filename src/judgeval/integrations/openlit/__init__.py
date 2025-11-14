from abc import ABC
from judgeval.tracer import Tracer
from judgeval.logger import judgeval_logger
from judgeval.utils.url import url_for
from judgeval.utils.project import _resolve_project_id


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
        tracer = Tracer.get_instance()
        if not tracer or not tracer._initialized:
            raise ValueError(
                "Openlit must be initialized after the tracer has been initialized. Please create the Tracer instance first before initializing Openlit."
            )

        api_key = tracer.api_key
        organization_id = tracer.organization_id
        project_name = tracer.project_name

        project_id = _resolve_project_id(project_name, api_key, organization_id)
        if not project_id:
            judgeval_logger.warning(
                f"Project {project_name} not found. Please create it first at https://app.judgmentlabs.ai/org/{organization_id}/projects."
            )
            return

        openlit.init(
            service_name=project_name,
            otlp_endpoint=url_for("/otel"),
            otlp_headers={
                "Authorization": f"Bearer {api_key}",
                "X-Organization-Id": organization_id,
                "X-Project-Id": project_id,
            },
            tracer=tracer.get_tracer(),
            **kwargs,
        )


__all__ = ["Openlit"]
