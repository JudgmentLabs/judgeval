from __future__ import annotations

from judgeval.v1.internal.api import JudgmentSyncClient


class TrainersFactory:
    __slots__ = "_client"

    def __init__(
        self,
        client: JudgmentSyncClient,
    ):
        self._client = client
