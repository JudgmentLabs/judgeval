from __future__ import annotations

from typing import List

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.datasets.dataset import Dataset, DatasetInfo
from judgeval.v1.data.example import Example


class DatasetFactory:
    __slots__ = "_client"

    def __init__(self, client: JudgmentSyncClient):
        self._client = client

    def get(self, name: str, project_name: str) -> Dataset:
        return Dataset.get(name, project_name, self._client)

    def create(
        self,
        name: str,
        project_name: str,
        examples: List[Example] = [],
        overwrite: bool = False,
    ) -> Dataset:
        return Dataset.create(name, project_name, examples, overwrite, self._client)

    def list(self, project_name: str) -> List[DatasetInfo]:
        return Dataset.list(project_name, self._client)
