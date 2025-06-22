from http import HTTPStatus
import os
from judgeval.common.api.client import APIClient
from judgeval.common.storage.storage import ABCStorage
from rich import print as rprint

from judgeval.common.tracer.utils import fallback_encoder


class JudgmentStorage(ABCStorage):
    """
    Abstract base class for storage systems, responsible for storing judgment data.
    """

    client: APIClient

    def __init__(self):
        super().__init__()

        JUDGMENT_API_KEY = os.environ.get("JUDGMENT_API_KEY")
        JUDGMENT_ORG_ID = os.environ.get("JUDGMENT_ORG_ID")

        if not JUDGMENT_API_KEY or not JUDGMENT_ORG_ID:
            raise ValueError(
                "Environment variables JUDGMENT_API_KEY and JUDGMENT_ORG_ID must be set."
            )

        self.client = APIClient(api_key=JUDGMENT_API_KEY, org_id=JUDGMENT_ORG_ID)

    def save_trace(self, trace_data, trace_id, project_name):
        # Placeholder implementation
        trace_json = trace_data.model_dump_json(fallback=fallback_encoder)
        response = self.client.do_post("/traces/save/", trace_json)

        if response.status_code == HTTPStatus.BAD_REQUEST:
            raise ValueError(
                f"Failed to save trace data: Check your Trace name for conflicts, set overwrite=True to overwrite existing traces: {response.text}"
            )
        elif response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to save trace data: {response.text}")

        server_response = response.json()
        if "ui_results_url" in server_response:
            pretty_str = f"\n🔍 You can view your trace data here: [rgb(106,0,255)][link={server_response['ui_results_url']}]View Trace[/link]\n"
            rprint(pretty_str)

        return server_response
