
from typing import Optional, List
import requests
from rich.progress import Progress, SpinnerColumn, TextColumn

from judgeval.common.logger import debug, error, warning, info
from judgeval.constants import (
    JUDGMENT_DATASETS_PUSH_API_URL,
    JUDGMENT_DATASETS_PULL_API_URL, 
    JUDGMENT_DATASETS_PULL_ALL_API_URL,
    JUDGMENT_DATASETS_EDIT_API_URL,
    JUDGMENT_DATASETS_EXPORT_JSONL_API_URL
)
from judgeval.data import Example, GroundTruthExample
from judgeval.data.datasets import EvalDataset




class EvalDatasetClient:
    def __init__(self, judgment_api_key: str, organization_id: str):
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id

    def create_dataset(self) -> EvalDataset:
        return EvalDataset(judgment_api_key=self.judgment_api_key)
    
    def push(self, dataset: EvalDataset, alias: str, overwrite: Optional[bool] = False) -> bool:
        debug(f"Pushing dataset with alias '{alias}' (overwrite={overwrite})")
        if overwrite:
            warning(f"Overwrite enabled for alias '{alias}'")
        """
        Pushes the dataset to Judgment platform

        Mock request:
        dataset = {
            "alias": alias,
            "ground_truths": [...],
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
                    "alias": alias,
                    "ground_truths": [g.to_dict() for g in dataset.ground_truths],
                    "examples": [e.to_dict() for e in dataset.examples],
                    "overwrite": overwrite,
                }
            try:
                response = requests.post(
                    JUDGMENT_DATASETS_PUSH_API_URL, 
                    json=content,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id
                    },
                    verify=True
                )
                if response.status_code == 500:
                    error(f"Server error during push: {content.get('message')}")
                    return False
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                if response.status_code == 422:
                    error(f"Validation error during push: {err.response.json()}")
                else:
                    error(f"HTTP error during push: {err}")
            
            info(f"Successfully pushed dataset with alias '{alias}'")
            payload = response.json()
            dataset._alias = payload.get("_alias")
            dataset._id = payload.get("_id")
            progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
                )
            return True
        
    def pull(self, alias: str) -> EvalDataset:
        debug(f"Pulling dataset with alias '{alias}'")
        """
        Pulls the dataset from Judgment platform

        Mock request:
        {
            "alias": alias,
            "user_id": user_id
        } 
        ==>
        {
            "ground_truths": [...],
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
                request_body = {
                    "alias": alias,
                }

                try:
                    response = requests.post(
                        JUDGMENT_DATASETS_PULL_API_URL, 
                        json=request_body,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.judgment_api_key}",
                            "X-Organization-Id": self.organization_id
                        },
                        verify=True
                    )
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    error(f"Error pulling dataset: {str(e)}")
                    raise

                info(f"Successfully pulled dataset with alias '{alias}'")
                payload = response.json()
                dataset.ground_truths = [GroundTruthExample(**g) for g in payload.get("ground_truths", [])]
                dataset.examples = [Example(**e) for e in payload.get("examples", [])]
                dataset._alias = payload.get("_alias")
                dataset._id = payload.get("_id")
                progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
                )

                return dataset

    def pull_all_user_dataset_stats(self) -> dict:
        debug(f"Pulling user datasets stats for user_id: {self.judgment_api_key}'")
        """
        Pulls the user datasets stats from Judgment platform

        Mock request:
        {
            "user_id": user_id
        } 
        ==>
        {
            "test_dataset_1": {"examples_count": len(dataset1.examples), "ground_truths_count": len(dataset1.ground_truths)},
            "test_dataset_2": {"examples_count": len(dataset2.examples), "ground_truths_count": len(dataset2.ground_truths)},
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
                    f"Pulling [rgb(106,0,255)]' datasets'[/rgb(106,0,255)] from Judgment...",
                    total=100,
                )
                request_body = {
                }

                try:
                    response = requests.post(
                        JUDGMENT_DATASETS_PULL_ALL_API_URL, 
                        json=request_body,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.judgment_api_key}",
                            "X-Organization-Id": self.organization_id
                        },
                        verify=True
                    )
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    error(f"Error pulling dataset: {str(e)}")
                    raise

                info(f"Successfully pulled datasets for userid: {self.judgment_api_key}'")
                payload = response.json()

                progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
                )

                return payload

    def edit_dataset(self, alias: str, examples: List[Example], ground_truths: List[GroundTruthExample]) -> bool:
        """
        Edits the dataset on Judgment platform by adding new examples and ground truths

        Mock request:
        {
            "alias": alias,
            "examples": [...],
            "ground_truths": [...],
            "judgment_api_key": self.judgment_api_key
        }
        """
        with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
            ) as progress:
            task_id = progress.add_task(
                f"Editing dataset [rgb(106,0,255)]'{alias}'[/rgb(106,0,255)] on Judgment...",
                total=100,
            )
            
            content = {
                "alias": alias,
                "examples": [e.to_dict() for e in examples],
                "ground_truths": [g.to_dict() for g in ground_truths],
            }

            try:
                response = requests.post(
                    JUDGMENT_DATASETS_EDIT_API_URL,
                    json=content,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id
                    },
                    verify=True
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                error(f"Error editing dataset: {str(e)}")
                return False
        
            info(f"Successfully edited dataset '{alias}'")
            return True

    def export_jsonl(self, alias: str) -> requests.Response:
        """Export dataset in JSONL format from Judgment platform"""
        debug(f"Exporting dataset with alias '{alias}' as JSONL")
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
                    json={"alias": alias},
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.judgment_api_key}",
                        "X-Organization-Id": self.organization_id
                    },
                    stream=True,
                    verify=True
                )
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    error(f"Dataset not found: {alias}")
                else:
                    error(f"HTTP error during export: {err}")
                raise
            except Exception as e:
                error(f"Error during export: {str(e)}")
                raise
                
            info(f"Successfully exported dataset with alias '{alias}'")
            progress.update(
                task_id,
                description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
            )
            
            return response
