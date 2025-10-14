from __future__ import annotations
from typing import TYPE_CHECKING

from judgeval.tracer.llm.llm_google.generate_content import (
    wrap_generate_content_sync,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from google.genai import Client


def wrap_google_client(tracer: Tracer, client: Client) -> Client:
    from judgeval.tracer.llm.llm_google.config import HAS_GOOGLE_GENAI
    from judgeval.logger import judgeval_logger

    if not HAS_GOOGLE_GENAI:
        judgeval_logger.error(
            "Cannot wrap Google GenAI client: 'google-genai' library not installed. "
            "Install it with: pip install google-genai"
        )
        return client

    from google.genai import Client

    if isinstance(client, Client):
        wrap_generate_content_sync(tracer, client)
        return client
    else:
        raise TypeError(f"Invalid client type: {type(client)}")
