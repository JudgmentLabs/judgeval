import datetime
import orjson
import os
import yaml
from dataclasses import dataclass
from typing import List, Literal, Optional

from judgeval.data import Example
from judgeval.data.trace import Trace
from judgeval.utils.file_utils import get_examples_from_yaml, get_examples_from_json
from judgeval.api import JudgmentSyncClient
from judgeval.logger import judgeval_logger
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID

from judgeval.data.judgment_types import DatasetKind


@dataclass
class DatasetInfo:
    dataset_id: str
    name: str
    created_at: str
    kind: DatasetKind
    entries: int
    creator: str


@dataclass
class Dataset:
    name: str
    project_name: str
    dataset_kind: DatasetKind = DatasetKind.example
    examples: Optional[List[Example]] = None
    traces: Optional[List[Trace]] = None
    judgment_api_key: str = JUDGMENT_API_KEY or ""
    organization_id: str = JUDGMENT_ORG_ID or ""

    @classmethod
    def get(
        cls,
        name: str,
        project_name: str,
    ):
        client = JudgmentSyncClient(cls.judgment_api_key, cls.organization_id)
        dataset = client.datasets_pull_for_judgeval(
            {
                "dataset_name": name,
                "project_name": project_name,
            },
        )
        if not dataset:
            raise ValueError(f"Dataset {name} not found in project {project_name}")

        dataset_kind = DatasetKind(dataset.get("dataset_kind", "example"))

        if dataset_kind == DatasetKind.example:
            examples = dataset.get("examples", [])
            if examples is None:
                examples = []

            for e in examples:
                if isinstance(e, dict) and isinstance(e.get("data", {}), dict):
                    e.update(e.pop("data"))  # type: ignore
                    e.pop(
                        "example_id"
                    )  # TODO: remove once scorer data migration is complete
            judgeval_logger.info(f"Successfully retrieved example dataset {name}!")
            return cls(
                name=name,
                project_name=project_name,
                dataset_kind=dataset_kind,
                examples=[Example(**e) for e in examples],
            )

        elif dataset_kind == DatasetKind.trace:
            trace_data = dataset.get("traces", [])
            if trace_data is None:
                trace_data = []

            traces = []
            for trace_item in trace_data:
                if isinstance(trace_item, dict):
                    trace = Trace.from_dataset_trace_with_spans(trace_item)
                    traces.append(trace)

            judgeval_logger.info(f"Successfully retrieved trace dataset {name}!")
            return cls(
                name=name,
                project_name=project_name,
                dataset_kind=dataset_kind,
                traces=traces,
            )

        else:
            raise ValueError(f"Unsupported dataset kind: {dataset_kind}")

    @classmethod
    def create(
        cls,
        name: str,
        project_name: str,
        examples: List[Example] = [],
        overwrite: bool = False,
    ):
        if not examples:
            examples = []

        client = JudgmentSyncClient(cls.judgment_api_key, cls.organization_id)
        client.datasets_create_for_judgeval(
            {
                "name": name,
                "project_name": project_name,
                "examples": examples,  # type: ignore
                "dataset_kind": "example",
                "overwrite": overwrite,
            }
        )

        judgeval_logger.info(f"Successfully created dataset {name}!")
        return cls(
            name=name,
            project_name=project_name,
            examples=examples,
        )

    @classmethod
    def list(cls, project_name: str):
        client = JudgmentSyncClient(cls.judgment_api_key, cls.organization_id)
        datasets = client.datasets_pull_all_for_judgeval({"project_name": project_name})

        judgeval_logger.info(f"Fetched all datasets for project {project_name}!")

        return [DatasetInfo(**dataset_info) for dataset_info in datasets]

    def add_from_json(self, file_path: str) -> None:
        """
        Adds examples from a JSON file.

        The JSON file is expected to have the following format:
        [
            {
                "key_01": "value_01",
                "key_02": "value_02"
            },
            {
                "key_11": "value_11",
                "key_12": "value_12",
                "key_13": "value_13"
            },
            ...
        ]
        """
        examples = get_examples_from_json(file_path)
        self.add_examples(examples)

    def add_from_yaml(self, file_path: str) -> None:
        """
        Adds examples from a YAML file.

        The YAML file is expected to have the following format:
        - key_01: value_01
          key_02: value_02
        - key_11: value_11
          key_12: value_12
          key_13: value_13
        ...
        """

        examples = get_examples_from_yaml(file_path)
        self.add_examples(examples)

    def add_examples(self, examples: List[Example]) -> None:
        if not isinstance(examples, list):
            raise TypeError("examples must be a list")

        client = JudgmentSyncClient(self.judgment_api_key, self.organization_id)
        client.datasets_insert_examples_for_judgeval(
            {
                "dataset_name": self.name,
                "project_name": self.project_name,
                "examples": examples,  # type: ignore
            }
        )

    def save_as(
        self,
        file_type: Literal["json", "yaml"],
        dir_path: str,
        save_name: str | None = None,
    ) -> None:
        """
        Saves the dataset as a file. Save only the examples.

        Args:
            file_type (Literal["json", "csv"]): The file type to save the dataset as.
            dir_path (str): The directory path to save the file to.
            save_name (str, optional): The name of the file to save. Defaults to None.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if save_name is None
            else save_name
        )
        complete_path = os.path.join(dir_path, f"{file_name}.{file_type}")
        if file_type == "json":
            with open(complete_path, "wb") as file:
                file.write(
                    orjson.dumps(
                        {
                            "examples": [e.to_dict() for e in self.examples]
                            if self.examples
                            else [],
                        },
                        option=orjson.OPT_INDENT_2,
                    )
                )
        elif file_type == "yaml":
            with open(complete_path, "w") as file:
                yaml_data = {
                    "examples": [e.to_dict() for e in self.examples]
                    if self.examples
                    else [],
                }
                yaml.dump(yaml_data, file, default_flow_style=False)
        else:
            ACCEPTABLE_FILE_TYPES = ["json", "yaml"]
            raise TypeError(
                f"Invalid file type: {file_type}. Please choose from {ACCEPTABLE_FILE_TYPES}"
            )

    def __iter__(self):
        if self.dataset_kind == DatasetKind.example and self.examples:
            return iter(self.examples)
        elif self.dataset_kind == DatasetKind.trace and self.traces:
            return iter(self.traces)
        else:
            return iter([])

    def __len__(self):
        if self.dataset_kind == DatasetKind.example and self.examples:
            return len(self.examples)
        elif self.dataset_kind == DatasetKind.trace and self.traces:
            return len(self.traces)
        else:
            return 0

    def __str__(self):
        if self.dataset_kind == DatasetKind.example:
            return (
                f"{self.__class__.__name__}(examples={self.examples}, name={self.name})"
            )
        else:
            return f"{self.__class__.__name__}(traces={self.traces}, name={self.name})"
