from __future__ import annotations

from typing import List

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.datasets.dataset import Dataset, DatasetInfo
from judgeval.v1.data.example import Example
from judgeval.logger import judgeval_logger


class DatasetFactory:
    __slots__ = "_client"

    def __init__(self, client: JudgmentSyncClient):
        self._client = client

    def get(self, name: str, project_name: str) -> Dataset:
        dataset = self._client.datasets_pull_for_judgeval(
            {
                "dataset_name": name,
                "project_name": project_name,
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
            project_name=project_name,
            dataset_kind=dataset_kind,
            examples=examples,
            client=self._client,
        )

    def create(
        self,
        name: str,
        project_name: str,
        examples: List[Example] = [],
        overwrite: bool = False,
    ) -> Dataset:
        self._client.datasets_create_for_judgeval(
            {
                "name": name,
                "project_name": project_name,
                "examples": [e.to_dict() for e in examples],
                "dataset_kind": "example",
                "overwrite": overwrite,
            }
        )

        judgeval_logger.info(f"Created dataset {name}")
        return Dataset(
            name=name, project_name=project_name, examples=examples, client=self._client
        )

    def list(self, project_name: str) -> List[DatasetInfo]:
        datasets = self._client.datasets_pull_all_for_judgeval(
            {"project_name": project_name}
        )
        judgeval_logger.info(f"Fetched datasets for project {project_name}")
        return [DatasetInfo(**d) for d in datasets]
