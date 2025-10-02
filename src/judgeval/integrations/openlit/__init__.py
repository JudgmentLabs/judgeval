from abc import ABC
from typing import Optional

from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID
from judgeval.tracer import Tracer
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_api_key, expect_organization_id
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
        project_name: str,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        /,
        **kwargs,
    ):
        tracer = Tracer.get_instance()
        if tracer and tracer._initialized:
            raise ValueError(
                "Openlit cannot be initialized after the tracer has been initialized. When using the Openlit integration, pass initialize=False to the Tracer constructor."
            )

        api_key = expect_api_key(api_key or JUDGMENT_API_KEY)
        organization_id = expect_organization_id(organization_id or JUDGMENT_ORG_ID)

        project_id = Tracer._resolve_project_id(project_name, api_key, organization_id)
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
            **kwargs,
        )


__all__ = ["Openlit"]
