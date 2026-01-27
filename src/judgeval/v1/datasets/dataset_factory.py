from __future__ import annotations

from typing import List, Iterable, Optional

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.datasets.dataset import Dataset, DatasetInfo
from judgeval.v1.data.example import Example
from judgeval.v1.utils import require_project_id
from judgeval.logger import judgeval_logger


class DatasetFactory:
    __slots__ = ("_client", "_default_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        default_project_id: Optional[str] = None,
        project_name: Optional[str] = None,
    ):
        self._client = client
        self._default_project_id = default_project_id
        self._project_name = project_name

    def get(self, name: str, project_id: Optional[str] = None) -> Dataset:
        pid = project_id or require_project_id(self._default_project_id)
        dataset = self._client.datasets_pull_for_judgeval(
            {
                "dataset_name": name,
                "project_id": pid,
            }
        )

        dataset_kind = dataset.get("dataset_kind", "example")
        examples_data = dataset.get("examples", []) or []

        examples = []
        for e in examples_data:
            if isinstance(e, dict):
                judgeval_logger.debug(f"Raw example keys: {e.keys()}")

                data_obj = e.get("data", {})
                if isinstance(data_obj, dict):
                    example_id = data_obj.get("example_id", "")
                    created_at = data_obj.get("created_at", "")
                    name_field = data_obj.get("name")

                    example = Example(
                        example_id=example_id, created_at=created_at, name=name_field
                    )

                    for key, value in data_obj.items():
                        if key not in ["example_id", "created_at", "name"]:
                            example.set_property(key, value)

                    examples.append(example)
                    judgeval_logger.debug(
                        f"Created example with name={name_field}, properties={list(example.properties.keys())}"
                    )

        judgeval_logger.info(f"Retrieved dataset {name} with {len(examples)} examples")
        return Dataset(
            name=name,
            project_id=pid,
            dataset_kind=dataset_kind,
            examples=examples,
            client=self._client,
            project_name=self._project_name or "",
        )

    def create(
        self,
        name: str,
        project_id: Optional[str] = None,
        examples: Iterable[Example] = [],
        overwrite: bool = False,
        batch_size: int = 100,
    ) -> Dataset:
        pid = project_id or require_project_id(self._default_project_id)
        self._client.datasets_create_for_judgeval(
            {
                "name": name,
                "project_id": pid,
                "examples": [],
                "dataset_kind": "example",
                "overwrite": overwrite,
            }
        )
        judgeval_logger.info(f"Created dataset {name}")

        if not isinstance(examples, list):
            examples = list(examples)

        dataset = Dataset(
            name=name,
            project_id=pid,
            examples=examples,
            client=self._client,
            project_name=self._project_name or "",
        )
        dataset.add_examples(examples, batch_size=batch_size)
        return dataset

    def list(self, project_id: Optional[str] = None) -> List[DatasetInfo]:
        pid = project_id or require_project_id(self._default_project_id)
        datasets = self._client.datasets_pull_all_for_judgeval({"project_id": pid})
        judgeval_logger.info(f"Fetched datasets for project {pid}")
        return [DatasetInfo(**d) for d in datasets]
