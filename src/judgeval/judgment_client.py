"""
Implements the JudgmentClient to interact with the Judgment API.
"""

import os
from uuid import uuid4
from typing import Optional, List, Dict, Any, Union, Callable
from requests import codes
from judgeval.utils.requests import requests
import asyncio

from judgeval.constants import ROOT_API
from judgeval.data.datasets import EvalDataset, EvalDatasetClient
from judgeval.data import (
    ScoringResult,
    Example,
    Trace,
)
from judgeval.scorers import (
    APIScorerConfig,
    BaseScorer,
    ClassifierScorer,
)
from judgeval.evaluation_run import EvaluationRun
from judgeval.run_evaluation import (
    run_eval,
    assert_test,
    run_trace_eval,
    safe_run_async,
)
from judgeval.data.trace_run import TraceRun
from judgeval.constants import (
    JUDGMENT_EVAL_FETCH_API_URL,
    JUDGMENT_PROJECT_DELETE_API_URL,
    JUDGMENT_PROJECT_CREATE_API_URL,
)
from judgeval.common.exceptions import JudgmentAPIError
from langchain_core.callbacks import BaseCallbackHandler
from judgeval.common.tracer import Tracer
from judgeval.common.utils import validate_api_key
from pydantic import BaseModel
from judgeval.run_evaluation import SpinnerWrappedTask
from judgeval.common.logger import judgeval_logger


class EvalRunRequestBody(BaseModel):
    eval_name: str
    project_name: str
    judgment_api_key: str


class DeleteEvalRunRequestBody(BaseModel):
    eval_names: List[str]
    project_name: str
    judgment_api_key: str


class SingletonMeta(type):
    _instances: Dict[type, "JudgmentClient"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class JudgmentClient(metaclass=SingletonMeta):
    def __init__(
        self,
        api_key: Optional[str] = os.getenv("JUDGMENT_API_KEY"),
        organization_id: Optional[str] = os.getenv("JUDGMENT_ORG_ID"),
    ):
        if not api_key:
            raise ValueError(
                "api_key parameter must be provided. Please provide a valid API key value or set the JUDGMENT_API_KEY environment variable."
            )

        if not organization_id:
            raise ValueError(
                "organization_id parameter must be provided. Please provide a valid organization ID value or set the JUDGMENT_ORG_ID environment variable."
            )

        self.judgment_api_key = api_key
        self.organization_id = organization_id
        self.eval_dataset_client = EvalDatasetClient(api_key, organization_id)

        # Verify API key is valid
        result, response = validate_api_key(api_key)
        if not result:
            # May be bad to output their invalid API key...
            raise JudgmentAPIError(f"Issue with passed in Judgment API key: {response}")
        else:
            judgeval_logger.info("Successfully initialized JudgmentClient!")

    def a_run_evaluation(
        self,
        examples: List[Example],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        model: Optional[str] = "gpt-4.1",
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_run",
        override: bool = False,
        append: bool = False,
    ) -> List[ScoringResult]:
        result = self.run_evaluation(
            examples=examples,
            scorers=scorers,
            model=model,
            project_name=project_name,
            eval_run_name=eval_run_name,
            override=override,
            append=append,
            async_execution=True,
        )
        assert not isinstance(result, (asyncio.Task, SpinnerWrappedTask))
        return result

    def run_trace_evaluation(
        self,
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        examples: Optional[List[Example]] = None,
        function: Optional[Callable] = None,
        tracer: Optional[Union[Tracer, BaseCallbackHandler]] = None,
        traces: Optional[List[Trace]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_trace",
        model: Optional[str] = "gpt-4.1",
        append: bool = False,
        override: bool = False,
    ) -> List[ScoringResult]:
        try:
            if examples and not function:
                raise ValueError("Cannot pass in examples without a function")

            if traces and function:
                raise ValueError("Cannot pass in traces and function")

            if examples and traces:
                raise ValueError("Cannot pass in both examples and traces")

            trace_run = TraceRun(
                project_name=project_name,
                eval_name=eval_run_name,
                traces=traces,
                scorers=scorers,
                model=model,
                append=append,
                judgment_api_key=self.judgment_api_key,
                organization_id=self.organization_id,
                tools=tools,
            )
            return run_trace_eval(trace_run, override, function, tracer, examples)
        except ValueError as e:
            raise ValueError(
                f"Please check your TraceRun object, one or more fields are invalid: \n{str(e)}"
            )
        except Exception as e:
            raise Exception(f"An unexpected error occurred during evaluation: {str(e)}")

    def run_evaluation(
        self,
        examples: List[Example],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        model: Optional[str] = "gpt-4.1",
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_run",
        override: bool = False,
        append: bool = False,
        async_execution: bool = False,
    ) -> Union[List[ScoringResult], asyncio.Task | SpinnerWrappedTask]:
        """
        Executes an evaluation of `Example`s using one or more `Scorer`s

        Args:
            examples (List[Example]): The examples to evaluate
            scorers (List[Union[APIScorerConfig, BaseScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
            override (bool): Whether to override an existing evaluation run with the same name
            append (bool): Whether to append to an existing evaluation run with the same name
            async_execution (bool): Whether to execute the evaluation asynchronously

        Returns:
            List[ScoringResult]: The results of the evaluation
        """
        if override and append:
            raise ValueError(
                "Cannot set both override and append to True. Please choose one."
            )

        try:
            eval = EvaluationRun(
                append=append,
                project_name=project_name,
                eval_name=eval_run_name,
                examples=examples,
                scorers=scorers,
                model=model,
                judgment_api_key=self.judgment_api_key,
                organization_id=self.organization_id,
            )
            return run_eval(
                eval,
                override,
                async_execution=async_execution,
            )
        except ValueError as e:
            raise ValueError(
                f"Please check your EvaluationRun object, one or more fields are invalid: \n{str(e)}"
            )
        except Exception as e:
            raise Exception(f"An unexpected error occurred during evaluation: {str(e)}")

    def create_dataset(self) -> EvalDataset:
        return self.eval_dataset_client.create_dataset()

    def push_dataset(
        self,
        alias: str,
        dataset: EvalDataset,
        project_name: str,
        overwrite: Optional[bool] = False,
    ) -> bool:
        """
        Uploads an `EvalDataset` to the Judgment platform for storage.

        Args:
            alias (str): The name to use for the dataset
            dataset (EvalDataset): The dataset to upload to Judgment
            overwrite (Optional[bool]): Whether to overwrite the dataset if it already exists

        Returns:
            bool: Whether the dataset was successfully uploaded
        """
        # Set judgment_api_key just in case it was not set
        dataset.judgment_api_key = self.judgment_api_key
        return self.eval_dataset_client.push(dataset, alias, project_name, overwrite)

    def append_dataset(
        self, alias: str, examples: List[Example], project_name: str
    ) -> bool:
        """
        Appends an `EvalDataset` to the Judgment platform for storage.
        """
        return self.eval_dataset_client.append_examples(alias, examples, project_name)

    def pull_dataset(self, alias: str, project_name: str) -> EvalDataset:
        """
        Retrieves a saved `EvalDataset` from the Judgment platform.

        Args:
            alias (str): The name of the dataset to retrieve

        Returns:
            EvalDataset: The retrieved dataset
        """
        return self.eval_dataset_client.pull(alias, project_name)

    def delete_dataset(self, alias: str, project_name: str) -> bool:
        """
        Deletes a saved `EvalDataset` from the Judgment platform.
        """
        return self.eval_dataset_client.delete(alias, project_name)

    def pull_project_dataset_stats(self, project_name: str) -> dict:
        """
        Retrieves all dataset stats from the Judgment platform for the project.

        Args:
            project_name (str): The name of the project to retrieve

        Returns:
            dict: The retrieved dataset stats
        """
        return self.eval_dataset_client.pull_project_dataset_stats(project_name)

    # Maybe add option where you can pass in the EvaluationRun object and it will pull the eval results from the backend
    def pull_eval(
        self, project_name: str, eval_run_name: str
    ) -> List[Dict[str, Union[str, List[ScoringResult]]]]:
        """Pull evaluation results from the server.

        Args:
            project_name (str): Name of the project
            eval_run_name (str): Name of the evaluation run

        Returns:
            Dict[str, Union[str, List[ScoringResult]]]: Dictionary containing:
                - id (str): The evaluation run ID
                - results (List[ScoringResult]): List of scoring results
        """
        eval_run_request_body = EvalRunRequestBody(
            project_name=project_name,
            eval_name=eval_run_name,
            judgment_api_key=self.judgment_api_key,
        )
        eval_run = requests.post(
            JUDGMENT_EVAL_FETCH_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id,
            },
            json=eval_run_request_body.model_dump(),
            verify=True,
        )
        if eval_run.status_code != codes.ok:
            raise ValueError(f"Error fetching eval results: {eval_run.json()}")

        return eval_run.json()

    def create_project(self, project_name: str) -> bool:
        """
        Creates a project on the server.
        """
        response = requests.post(
            JUDGMENT_PROJECT_CREATE_API_URL,
            json={
                "project_name": project_name,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id,
            },
        )
        if response.status_code != codes.ok:
            raise ValueError(f"Error creating project: {response.json()}")
        return response.json()

    def delete_project(self, project_name: str) -> bool:
        """
        Deletes a project from the server. Which also deletes all evaluations and traces associated with the project.
        """
        response = requests.delete(
            JUDGMENT_PROJECT_DELETE_API_URL,
            json={
                "project_name": project_name,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id,
            },
        )
        if response.status_code != codes.ok:
            raise ValueError(f"Error deleting project: {response.json()}")
        return response.json()

    def fetch_classifier_scorer(self, slug: str) -> ClassifierScorer:
        """
        Fetches a classifier scorer configuration from the Judgment API.

        Args:
            slug (str): Slug identifier of the custom scorer to fetch

        Returns:
            ClassifierScorer: The configured classifier scorer object

        Raises:
            JudgmentAPIError: If the scorer cannot be fetched or doesn't exist
        """
        request_body = {
            "slug": slug,
        }

        response = requests.post(
            f"{ROOT_API}/fetch_scorer/",
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id,
            },
            verify=True,
        )

        if response.status_code == 500:
            raise JudgmentAPIError(
                f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {response.json().get('detail', '')}"
            )
        elif response.status_code != 200:
            raise JudgmentAPIError(
                f"Failed to fetch classifier scorer '{slug}': {response.json().get('detail', '')}"
            )

        scorer_config = response.json()
        scorer_config.pop("created_at")
        scorer_config.pop("updated_at")

        try:
            return ClassifierScorer(**scorer_config)
        except Exception as e:
            raise JudgmentAPIError(
                f"Failed to create classifier scorer '{slug}' with config {scorer_config}: {str(e)}"
            )

    def push_classifier_scorer(
        self, scorer: ClassifierScorer, slug: str | None = None
    ) -> str:
        """
        Pushes a classifier scorer configuration to the Judgment API.

        Args:
            slug (str): Slug identifier for the scorer. If it exists, the scorer will be updated.
            scorer (ClassifierScorer): The classifier scorer to save

        Returns:
            str: The slug identifier of the saved scorer

        Raises:
            JudgmentAPIError: If there's an error saving the scorer
        """
        request_body = {
            "name": scorer.name,
            "conversation": scorer.conversation,
            "options": scorer.options,
            "slug": slug,
        }

        response = requests.post(
            f"{ROOT_API}/save_scorer/",
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id,
            },
            verify=True,
        )

        if response.status_code == 500:
            raise JudgmentAPIError(
                f"The server is temporarily unavailable. \
                                   Please try your request again in a few moments. \
                                   Error details: {response.json().get('detail', '')}"
            )
        elif response.status_code != 200:
            raise JudgmentAPIError(
                f"Failed to save classifier scorer: {response.json().get('detail', '')}"
            )

        return response.json()["slug"]

    def assert_test(
        self,
        examples: List[Example],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        model: Optional[str] = "gpt-4.1",
        project_name: str = "default_test",
        eval_run_name: str = str(uuid4()),
        override: bool = False,
        append: bool = False,
        async_execution: bool = False,
    ) -> None:
        """
        Asserts a test by running the evaluation and checking the results for success

        Args:
            examples (List[Example]): The examples to evaluate.
            scorers (List[Union[APIScorerConfig, BaseScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
            override (bool): Whether to override an existing evaluation run with the same name
            append (bool): Whether to append to an existing evaluation run with the same name
            async_execution (bool): Whether to run the evaluation asynchronously
        """

        results: Union[List[ScoringResult], asyncio.Task | SpinnerWrappedTask]

        results = self.run_evaluation(
            examples=examples,
            scorers=scorers,
            model=model,
            project_name=project_name,
            eval_run_name=eval_run_name,
            override=override,
            append=append,
            async_execution=async_execution,
        )

        if async_execution and isinstance(results, (asyncio.Task, SpinnerWrappedTask)):

            async def run_async():  # Using wrapper here to resolve mypy error with passing Task into asyncio.run
                return await results

            actual_results = safe_run_async(run_async())
            assert_test(actual_results)  # Call the synchronous imported function
        else:
            # 'results' is already List[ScoringResult] here (synchronous path)
            assert_test(results)  # Call the synchronous imported function

    def assert_trace_test(
        self,
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        examples: Optional[List[Example]] = None,
        function: Optional[Callable] = None,
        tracer: Optional[Union[Tracer, BaseCallbackHandler]] = None,
        traces: Optional[List[Trace]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = "gpt-4.1",
        project_name: str = "default_test",
        eval_run_name: str = str(uuid4()),
        override: bool = False,
        append: bool = False,
        async_execution: bool = False,
    ) -> None:
        """
        Asserts a test by running the evaluation and checking the results for success

        Args:
            examples (List[Example]): The examples to evaluate.
            scorers (List[Union[APIScorerConfig, BaseScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
            override (bool): Whether to override an existing evaluation run with the same name
            append (bool): Whether to append to an existing evaluation run with the same name
            function (Optional[Callable]): A function to use for evaluation
            tracer (Optional[Union[Tracer, BaseCallbackHandler]]): A tracer to use for evaluation
            tools (Optional[List[Dict[str, Any]]]): A list of tools to use for evaluation
            async_execution (bool): Whether to run the evaluation asynchronously
        """

        # Check for enable_param_checking and tools
        for scorer in scorers:
            if hasattr(scorer, "kwargs") and scorer.kwargs is not None:
                if scorer.kwargs.get("enable_param_checking") is True:
                    if not tools:
                        raise ValueError(
                            f"You must provide the 'tools' argument to assert_test when using a scorer with enable_param_checking=True. If you do not want to do param checking, explicitly set enable_param_checking=False for the {scorer.__name__} scorer."
                        )

        results: Union[List[ScoringResult], asyncio.Task | SpinnerWrappedTask]

        results = self.run_trace_evaluation(
            examples=examples,
            traces=traces,
            scorers=scorers,
            model=model,
            project_name=project_name,
            eval_run_name=eval_run_name,
            override=override,
            append=append,
            function=function,
            tracer=tracer,
            tools=tools,
        )

        if async_execution and isinstance(results, (asyncio.Task, SpinnerWrappedTask)):

            async def run_async():  # Using wrapper here to resolve mypy error with passing Task into asyncio.run
                return await results

            actual_results = safe_run_async(run_async())
            assert_test(actual_results)  # Call the synchronous imported function
        else:
            # 'results' is already List[ScoringResult] here (synchronous path)
            assert_test(results)  # Call the synchronous imported function
