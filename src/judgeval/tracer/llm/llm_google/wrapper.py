from __future__ import annotations
from typing import TYPE_CHECKING

from judgeval.tracer.llm.llm_google.generate_content import (
    wrap_generate_content_sync,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from google.genai import Client


def wrap_google_client(tracer: Tracer, client: Client) -> Client:
    from judgeval.tracer.llm.llm_google.config import google_genai_Client

    if google_genai_Client is not None and isinstance(client, google_genai_Client):
        wrap_generate_content_sync(tracer, client)
        return client
    else:
        raise TypeError(f"Invalid client type: {type(client)}")
