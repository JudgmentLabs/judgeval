from __future__ import annotations

from typing import Any, Callable, Optional

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
        serializer: Optional[Callable[[Any], str]] = None,
        initialize: bool = True,
        **kwargs: Any,
    ) -> Tracer:
        if serializer is None:
            serializer = safe_serialize

        return Tracer(
            project_name=project_name,
            enable_evaluation=enable_evaluation,
            api_client=self._client,
            serializer=serializer,
            initialize=initialize,
        )
