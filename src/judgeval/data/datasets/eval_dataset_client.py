from typing import Optional, List
from requests import Response, exceptions
from judgeval.utils.requests import requests
from rich.progress import Progress, SpinnerColumn, TextColumn
from judgeval.common.logger import judgeval_logger
from judgeval.constants import (
    JUDGMENT_DATASETS_PUSH_API_URL,
    JUDGMENT_DATASETS_APPEND_EXAMPLES_API_URL,
    JUDGMENT_DATASETS_PULL_API_URL,
    JUDGMENT_DATASETS_PROJECT_STATS_API_URL,
    JUDGMENT_DATASETS_DELETE_API_URL,
    JUDGMENT_DATASETS_EXPORT_JSONL_API_URL,
)
from judgeval.data import Example, Trace
from judgeval.data.datasets import EvalDataset


class EvalDatasetClient:
    def __init__(self, judgment_api_key: str, organization_id: str):
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id

    def create_dataset(self) -> EvalDataset:
        return EvalDataset(judgment_api_key=self.judgment_api_key)

    def push(
        self,
        dataset: EvalDataset,
        alias: str,
        project_name: str,
        overwrite: Optional[bool] = False,
    ) -> bool:
        if overwrite:
            judgeval_logger.warning(f"Overwrite enabled for alias '{alias}'")
        """
        Pushes the dataset to Judgment platform

        Mock request:
        dataset = {
            "alias": alias,
            "examples": [...],
            "overwrite": overwrite
        } ==>
        {
            "_alias": alias,
            "_id": "..."  # ID of the dataset
        }
        """
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"Pushing [rgb(106,0,255)]'{alias}' to Judgment...",
                total=100,
            )
            content = {
                "dataset_alias": alias,
                "project_name": project_name,
                "examples": [e.to_dict() for e in dataset.examples],
                "traces": [t.model_dump() for t in dataset.traces],
                "overwrite": overwrite,
            }
            try:
                response = requests.post(
                    JUDGMENT_DATASETS_PUSH_API_URL,
                    json=content,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id,
                    },
                    verify=True,
                )
                if response.status_code != 200:
                    judgeval_logger.error(
                        f"Server error during push: {response.json()}"
                    )
                    raise Exception(f"Server error during push: {response.json()}")
                response.raise_for_status()
            except exceptions.HTTPError as err:
                if response.status_code == 422:
                    judgeval_logger.error(
                        f"Validation error during push: {err.response.json()}"
                    )
                else:
                    judgeval_logger.error(f"HTTP error during push: {err}")

            payload = response.json()
            dataset._alias = payload.get("_alias")
            dataset._id = payload.get("_id")
            progress.update(
                task_id,
                description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
            )
            return True

    def append_examples(
        self, alias: str, examples: List[Example], project_name: str
    ) -> bool:
        """
        Appends the dataset to Judgment platform

        Mock request:
        dataset = {
            "alias": alias,
            "examples": [...],
            "project_name": project_name
        } ==>
        {
            "_alias": alias,
            "_id": "..."  # ID of the dataset
        }
        """
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"Appending [rgb(106,0,255)]'{alias}' to Judgment...",
                total=100,
            )
            content = {
                "dataset_alias": alias,
                "project_name": project_name,
                "examples": [e.to_dict() for e in examples],
            }
            try:
                response = requests.post(
                    JUDGMENT_DATASETS_APPEND_EXAMPLES_API_URL,
                    json=content,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id,
                    },
                    verify=True,
                )
                if response.status_code != 200:
                    judgeval_logger.error(
                        f"Server error during append: {response.json()}"
                    )
                    raise Exception(f"Server error during append: {response.json()}")
                response.raise_for_status()
            except exceptions.HTTPError as err:
                if response.status_code == 422:
                    judgeval_logger.error(
                        f"Validation error during append: {err.response.json()}"
                    )
                else:
                    judgeval_logger.error(f"HTTP error during append: {err}")

            progress.update(
                task_id,
                description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
            )
            return True

    def pull(self, alias: str, project_name: str) -> EvalDataset:
        """
        Pulls the dataset from Judgment platform

        Mock request:
        {
            "alias": alias,
            "project_name": project_name
        }
        ==>
        {
            "examples": [...],
            "_alias": alias,
            "_id": "..."  # ID of the dataset
        }
        """
        # Make a POST request to the Judgment API to get the dataset
        dataset = self.create_dataset()

        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"Pulling [rgb(106,0,255)]'{alias}'[/rgb(106,0,255)] from Judgment...",
                total=100,
            )
            request_body = {"dataset_alias": alias, "project_name": project_name}

            try:
                response = requests.post(
                    JUDGMENT_DATASETS_PULL_API_URL,
                    json=request_body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id,
                    },
                    verify=True,
                )
                response.raise_for_status()
            except exceptions.RequestException as e:
                judgeval_logger.error(f"Error pulling dataset: {str(e)}")
                raise

            payload = response.json()
            dataset.examples = [Example(**e) for e in payload.get("examples", [])]
            dataset.traces = [Trace(**t) for t in payload.get("traces", [])]
            dataset._alias = payload.get("alias")
            dataset._id = payload.get("id")
            progress.update(
                task_id,
                description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
            )

            return dataset

    def delete(self, alias: str, project_name: str) -> bool:
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            progress.add_task(
                f"Deleting [rgb(106,0,255)]'{alias}'[/rgb(106,0,255)] from Judgment...",
                total=100,
            )
            request_body = {"dataset_alias": alias, "project_name": project_name}

            try:
                response = requests.post(
                    JUDGMENT_DATASETS_DELETE_API_URL,
                    json=request_body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id,
                    },
                    verify=True,
                )
                response.raise_for_status()
            except exceptions.RequestException as e:
                judgeval_logger.error(f"Error deleting dataset: {str(e)}")
                raise

            return True

    def pull_project_dataset_stats(self, project_name: str) -> dict:
        """
        Pulls the project datasets stats from Judgment platform

        Mock request:
        {
            "project_name": project_name
        }
        ==>
        {
            "test_dataset_1": {"examples_count": len(dataset1.examples)},
            "test_dataset_2": {"examples_count": len(dataset2.examples)},
            ...
        }
        """
        # Make a POST request to the Judgment API to get the dataset

        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                "Pulling [rgb(106,0,255)]' datasets'[/rgb(106,0,255)] from Judgment...",
                total=100,
            )
            request_body = {"project_name": project_name}

            try:
                response = requests.post(
                    JUDGMENT_DATASETS_PROJECT_STATS_API_URL,
                    json=request_body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id,
                    },
                    verify=True,
                )
                response.raise_for_status()
            except exceptions.RequestException as e:
                judgeval_logger.error(f"Error pulling dataset: {str(e)}")
                raise

            payload = response.json()

            progress.update(
                task_id,
                description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
            )

            return payload

    def export_jsonl(self, alias: str, project_name: str) -> Response:
        """Export dataset in JSONL format from Judgment platform"""
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"Exporting [rgb(106,0,255)]'{alias}'[/rgb(106,0,255)] as JSONL...",
                total=100,
            )
            try:
                response = requests.post(
                    JUDGMENT_DATASETS_EXPORT_JSONL_API_URL,
                    json={"dataset_alias": alias, "project_name": project_name},
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id,
                    },
                    stream=True,
                    verify=True,
                )
                response.raise_for_status()
            except exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    judgeval_logger.error(f"Dataset not found: {alias}")
                else:
                    judgeval_logger.error(f"HTTP error during export: {err}")
                raise
            except Exception as e:
                judgeval_logger.error(f"Error during export: {str(e)}")
                raise

            progress.update(
                task_id,
                description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
            )

            return response
