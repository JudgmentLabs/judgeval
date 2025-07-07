from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from judgeval.constants import (
    JUDGMENT_TRACES_UPSERT_API_URL,
)
from judgeval.utils.requests import requests

if TYPE_CHECKING:
    from judgeval.common.tracer import Tracer


from rich import print as rprint

import json
import warnings
from http import HTTPStatus


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
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id
        self.tracer = tracer

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

        def fallback_encoder(obj):
            """
            Custom JSON encoder fallback.
            Tries to use obj.__repr__(), then str(obj) if that fails or for a simpler string.
            """
            try:
                return repr(obj)
            except Exception:
                try:
                    return str(obj)
                except Exception as e:
                    return f"<Unserializable object of type {type(obj).__name__}: {e}>"

        serialized_trace_data = json.dumps(trace_data, default=fallback_encoder)

        response = requests.post(
            JUDGMENT_TRACES_UPSERT_API_URL,
            data=serialized_trace_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id,
            },
            verify=True,
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to upsert trace data: {response.text}")

        # Parse server response
        server_response = response.json()

        # If S3 storage is enabled, save to S3 only on final save
        if self.tracer and self.tracer.use_s3 and final_save:
            try:
                s3_key = self.tracer.s3_storage.save_trace(
                    trace_data=trace_data,
                    trace_id=trace_data["trace_id"],
                    project_name=trace_data["project_name"],
                )
                print(f"Trace also saved to S3 at key: {s3_key}")
            except Exception as e:
                warnings.warn(f"Failed to save trace to S3: {str(e)}")

        if not offline_mode and show_link and "ui_results_url" in server_response:
            pretty_str = f"\n🔍 You can view your trace data here: [rgb(106,0,255)][link={server_response['ui_results_url']}]View Trace[/link]\n"
            rprint(pretty_str)

        return server_response
