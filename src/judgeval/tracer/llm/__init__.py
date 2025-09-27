from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from judgeval.tracer.llm.providers import ApiClient
    from judgeval.tracer import Tracer


def wrap_provider(tracer: Tracer, client: ApiClient) -> ApiClient:
    return client
