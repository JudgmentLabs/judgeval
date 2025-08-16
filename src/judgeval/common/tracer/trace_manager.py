from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from judgeval.common.tracer import Tracer

from judgeval.common.logger import judgeval_logger
from judgeval.common.api import JudgmentApiClient
from rich import print as rprint


class TraceManagerClient:
    """
    Client for handling trace endpoints with the Judgment API


    Operations include:
    - Fetching a trace by id
    - Saving a trace
    - Deleting a trace
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
        tracer: Optional[Tracer] = None,
    ):
        self.api_client = JudgmentApiClient(judgment_api_key, organization_id)
        self.tracer = tracer

    def fetch_trace(self, trace_id: str):
        """
        Fetch a trace by its id
        """
        return self.api_client.fetch_trace(trace_id)

    def upsert_trace(
        self,
        trace_data: dict,
        offline_mode: bool = False,
        show_link: bool = True,
        final_save: bool = True,
    ):
        """
        Upserts a trace to the Judgment API (always overwrites if exists).

        Args:
            trace_data: The trace data to upsert
            offline_mode: Whether running in offline mode
            show_link: Whether to show the UI link (for live tracing)
            final_save: Whether this is the final save (controls S3 saving)

        Returns:
            dict: Server response containing UI URL and other metadata
        """

        if self.tracer and self.tracer.use_s3 and final_save:
            try:
                s3_key = self.tracer.s3_storage.save_trace(
                    trace_data=trace_data,
                    trace_id=trace_data["trace_id"],
                    project_name=trace_data["project_name"],
                )
                judgeval_logger.info(f"Trace also saved to S3 at key: {s3_key}")
            except Exception as e:
                judgeval_logger.warning(f"Failed to save trace to S3: {str(e)}")

        trace_data.pop("trace_spans", None)
        trace_data.pop("evaluation_runs", None)

        server_response = self.api_client.upsert_trace(trace_data)

        if (
            not offline_mode
            and show_link
            and "ui_results_url" in server_response
            and self.tracer.show_trace_urls
        ):
            pretty_str = f"\n🔍 You can view your trace data here: [rgb(106,0,255)][link={server_response['ui_results_url']}]View Trace[/link]\n"
            rprint(pretty_str)

        return server_response

    def delete_trace(self, trace_id: str):
        """
        Delete a trace from the database.
        """
        return self.api_client.delete_trace(trace_id)

    def delete_traces(self, trace_ids: List[str]):
        """
        Delete a batch of traces from the database.
        """
        return self.api_client.delete_traces(trace_ids)

    def delete_project(self, project_name: str):
        """
        Deletes a project from the server. Which also deletes all evaluations and traces associated with the project.
        """
        return self.api_client.delete_project(project_name)
