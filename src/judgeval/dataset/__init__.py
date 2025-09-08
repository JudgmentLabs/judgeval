import datetime
import orjson
import os
import yaml
from dataclasses import dataclass
from typing import List, Literal, Optional

from judgeval.data import Example
from judgeval.utils.file_utils import get_examples_from_yaml, get_examples_from_json
from judgeval.api import JudgmentSyncClient
from judgeval.logger import judgeval_logger
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID


@dataclass
class Dataset:
    examples: List[Example]
    name: str
    project_name: str
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
        examples = dataset.get("examples", [])
        for e in examples:
            if isinstance(e, dict) and isinstance(e.get("data"), dict):
                e.update(e.pop("data"))
                e.pop(
                    "example_id"
                )  # TODO: remove once scorer data migraiton is complete
        judgeval_logger.info(f"Succesfully retrieved dataset {name}!")
        return cls(
            name=name,
            project_name=project_name,
            examples=[Example(**e) for e in examples],
        )

    @classmethod
    def create(
        cls,
        name: str,
        project_name: str,
        examples: Optional[List[Example]] = None,
        overwrite: bool = False,
    ):
        if not examples:
            examples = []

        client = JudgmentSyncClient(cls.judgment_api_key, cls.organization_id)
        client.datasets_create_for_judgeval(
            {
                "name": name,
                "project_name": project_name,
                "examples": [e.model_dump() for e in examples],
                "dataset_kind": "example",
                "overwrite": overwrite,
            }
        )

        judgeval_logger.info(f"Succesfull created dataset {name}!")
        return cls(
            name=name,
            project_name=project_name,
            examples=examples,
        )

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
        client = JudgmentSyncClient(self.judgment_api_key, self.organization_id)
        client.datasets_insert_examples_for_judgeval(
            {
                "dataset_name": self.name,
                "project_name": self.project_name,
                "examples": [e.model_dump() for e in examples],
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
                            "examples": [e.to_dict() for e in self.examples],
                        },
                        option=orjson.OPT_INDENT_2,
                    )
                )
        elif file_type == "yaml":
            with open(complete_path, "w") as file:
                yaml_data = {
                    "examples": [e.to_dict() for e in self.examples],
                }
                yaml.dump(yaml_data, file, default_flow_style=False)
        else:
            ACCEPTABLE_FILE_TYPES = ["json", "yaml"]
            raise TypeError(
                f"Invalid file type: {file_type}. Please choose from {ACCEPTABLE_FILE_TYPES}"
            )

    def __iter__(self):
        return iter(self.examples)

    def __len__(self):
        return len(self.examples)

    def __str__(self):
        return f"{self.__class__.__name__}(examples={self.examples}, name={self.name})"
