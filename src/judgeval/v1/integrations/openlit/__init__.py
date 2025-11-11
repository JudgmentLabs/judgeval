from abc import ABC
from judgeval.v1.tracer import Tracer
from judgeval.logger import judgeval_logger
from judgeval.utils.url import url_for
from judgeval.v1.utils import resolve_project_id


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
        api_key = tracer.api_client.api_key
        organization_id = tracer.api_client.organization_id
        project_name = tracer.project_name

        project_id = resolve_project_id(tracer.api_client, project_name)
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
            disable_metrics=True,
            **kwargs,
        )


__all__ = ["Openlit"]
