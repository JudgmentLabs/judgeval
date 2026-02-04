from __future__ import annotations

from typing import List, Iterable, Optional

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.datasets.dataset import Dataset, DatasetInfo
from judgeval.v1.data.example import Example
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id


class DatasetFactory:
    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    def get(self, name: str) -> Optional[Dataset]:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None
        dataset = self._client.get_projects_datasets_by_dataset_name(
            project_id=project_id,
            dataset_name=name,
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
            project_id=project_id,
            dataset_kind=dataset_kind,
            examples=examples,
            client=self._client,
            project_name=self._project_name,
        )

    def create(
        self,
        name: str,
        examples: Iterable[Example] = [],
        overwrite: bool = False,
        batch_size: int = 100,
    ) -> Optional[Dataset]:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        self._client.post_projects_datasets(
            project_id=project_id,
            payload={
                "name": name,
                "examples": [],
                "dataset_kind": "example",
                "overwrite": overwrite,
            },
        )
        judgeval_logger.info(f"Created dataset {name}")

        if not isinstance(examples, list):
            examples = list(examples)

        dataset = Dataset(
            name=name,
            project_id=project_id,
            examples=examples,
            client=self._client,
            project_name=self._project_name,
        )
        dataset.add_examples(examples, batch_size=batch_size)
        return dataset

    def list(self) -> Optional[List[DatasetInfo]]:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        datasets = self._client.get_projects_datasets(
            project_id=project_id,
        )
        judgeval_logger.info(f"Fetched datasets for project {project_id}")
        return [DatasetInfo(**d) for d in datasets]
