from __future__ import annotations

from typing import Any, Callable

from judgeval.utils.serialize import safe_serialize
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.tracer.tracer import Tracer


class TracerFactory:
    __slots__ = "_client"

    def __init__(
        self,
        client: JudgmentSyncClient,
    ):
        self._client = client

    def create(
        self,
        project_name: str,
        enable_evaluation: bool = True,
        serializer: Callable[[Any], str] = safe_serialize,
        initialize: bool = True,
    ) -> Tracer:
        return Tracer(
            project_name=project_name,
            enable_evaluation=enable_evaluation,
            api_client=self._client,
            serializer=serializer,
            initialize=initialize,
        )
