"""
Implements the JudgmentClient to interact with the Judgment API.
"""

from __future__ import annotations
import os
import importlib.util
from pathlib import Path
from uuid import uuid4
from typing import Optional, List, Dict, Any, Union, Callable, TYPE_CHECKING

from judgeval.data import (
    ScoringResult,
    Example,
    Trace,
)
from judgeval.scorers import (
    APIScorerConfig,
    BaseScorer,
)
from judgeval.data.evaluation_run import EvaluationRun
from judgeval.run_evaluation import (
    run_eval,
    assert_test,
    run_trace_eval,
)
from judgeval.data.trace_run import TraceRun
from judgeval.common.api import JudgmentApiClient
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.common.tracer import Tracer
from judgeval.common.utils import validate_api_key
from pydantic import BaseModel
from judgeval.common.logger import judgeval_logger


if TYPE_CHECKING:
    from judgeval.integrations.langgraph import JudgevalCallbackHandler
from judgeval.constants import DEFAULT_GPT_MODEL


class EvalRunRequestBody(BaseModel):
    eval_name: str
    project_name: str


class DeleteEvalRunRequestBody(BaseModel):
    eval_names: List[str]
    project_name: str


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
        self.api_client = JudgmentApiClient(api_key, organization_id)

        # Verify API key is valid
        result, response = validate_api_key(api_key)
        if not result:
            # May be bad to output their invalid API key...
            raise JudgmentAPIError(f"Issue with passed in Judgment API key: {response}")
        else:
            judgeval_logger.info("Successfully initialized JudgmentClient!")

    def run_trace_evaluation(
        self,
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        examples: Optional[List[Example]] = None,
        function: Optional[Callable] = None,
        tracer: Optional[Union[Tracer, JudgevalCallbackHandler]] = None,
        traces: Optional[List[Trace]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_trace",
        model: Optional[str] = DEFAULT_GPT_MODEL,
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
                organization_id=self.organization_id,
                tools=tools,
            )
            return run_trace_eval(
                trace_run, self.judgment_api_key, function, tracer, examples
            )
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
        model: Optional[str] = DEFAULT_GPT_MODEL,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_run",
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of `Example`s using one or more `Scorer`s

        Args:
            examples (List[Example]): The examples to evaluate
            scorers (List[Union[APIScorerConfig, BaseScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run

        Returns:
            List[ScoringResult]: The results of the evaluation
        """

        try:
            eval = EvaluationRun(
                project_name=project_name,
                eval_name=eval_run_name,
                examples=examples,
                scorers=scorers,
                model=model,
                organization_id=self.organization_id,
            )
            return run_eval(
                eval,
                self.judgment_api_key,
            )
        except ValueError as e:
            raise ValueError(
                f"Please check your EvaluationRun object, one or more fields are invalid: \n{str(e)}"
            )
        except Exception as e:
            raise Exception(f"An unexpected error occurred during evaluation: {str(e)}")

    def create_project(self, project_name: str) -> bool:
        """
        Creates a project on the server.
        """
        try:
            self.api_client.create_project(project_name)
            return True
        except Exception as e:
            judgeval_logger.error(f"Error creating project: {e}")
            return False

    def delete_project(self, project_name: str) -> bool:
        """
        Deletes a project from the server. Which also deletes all evaluations and traces associated with the project.
        """
        self.api_client.delete_project(project_name)
        return True

    def assert_test(
        self,
        examples: List[Example],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        model: Optional[str] = DEFAULT_GPT_MODEL,
        project_name: str = "default_test",
        eval_run_name: str = str(uuid4()),
    ) -> None:
        """
        Asserts a test by running the evaluation and checking the results for success

        Args:
            examples (List[Example]): The examples to evaluate.
            scorers (List[Union[APIScorerConfig, BaseScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
        """

        results: List[ScoringResult]

        results = self.run_evaluation(
            examples=examples,
            scorers=scorers,
            model=model,
            project_name=project_name,
            eval_run_name=eval_run_name,
        )
        assert_test(results)

    def assert_trace_test(
        self,
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        examples: Optional[List[Example]] = None,
        function: Optional[Callable] = None,
        tracer: Optional[Union[Tracer, JudgevalCallbackHandler]] = None,
        traces: Optional[List[Trace]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = DEFAULT_GPT_MODEL,
        project_name: str = "default_test",
        eval_run_name: str = str(uuid4()),
    ) -> None:
        """
        Asserts a test by running the evaluation and checking the results for success

        Args:
            examples (List[Example]): The examples to evaluate.
            scorers (List[Union[APIScorerConfig, BaseScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
            function (Optional[Callable]): A function to use for evaluation
            tracer (Optional[Union[Tracer, BaseCallbackHandler]]): A tracer to use for evaluation
            tools (Optional[List[Dict[str, Any]]]): A list of tools to use for evaluation
        """

        # Check for enable_param_checking and tools
        for scorer in scorers:
            if hasattr(scorer, "kwargs") and scorer.kwargs is not None:
                if scorer.kwargs.get("enable_param_checking") is True:
                    if not tools:
                        raise ValueError(
                            f"You must provide the 'tools' argument to assert_test when using a scorer with enable_param_checking=True. If you do not want to do param checking, explicitly set enable_param_checking=False for the {scorer.__name__} scorer."
                        )

        results: List[ScoringResult]

        results = self.run_trace_evaluation(
            examples=examples,
            traces=traces,
            scorers=scorers,
            model=model,
            project_name=project_name,
            eval_run_name=eval_run_name,
            function=function,
            tracer=tracer,
            tools=tools,
        )

        assert_test(results)

    def _extract_scorer_name(self, scorer_file_path: str) -> str:
        """Extract scorer name from the scorer file by importing it."""
        try:
            spec = importlib.util.spec_from_file_location(
                "scorer_module", scorer_file_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec from {scorer_file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and any("Scorer" in str(base) for base in attr.__mro__)
                    and attr.__module__ == "scorer_module"
                ):
                    try:
                        # Instantiate the scorer and get its name
                        scorer_instance = attr()
                        if hasattr(scorer_instance, "name"):
                            return scorer_instance.name
                    except Exception:
                        # Skip if instantiation fails
                        continue

            raise AttributeError("No scorer class found or could be instantiated")
        except Exception as e:
            judgeval_logger.warning(f"Could not extract scorer name: {e}")
            return Path(scorer_file_path).stem

    def upload_custom_scorer(
        self,
        scorer_file_path: str,
        requirements_file_path: Optional[str] = None,
        unique_name: Optional[str] = None,
    ) -> bool:
        """
        Upload custom ExampleScorer from files to backend.

        Args:
            scorer_file_path: Path to Python file containing CustomScorer class
            requirements_file_path: Optional path to requirements.txt
            unique_name: Optional unique identifier (auto-detected from scorer.name if not provided)

        Returns:
            bool: True if upload successful

        Raises:
            ValueError: If scorer file is invalid
            FileNotFoundError: If scorer file doesn't exist
        """
        import os

        if not os.path.exists(scorer_file_path):
            raise FileNotFoundError(f"Scorer file not found: {scorer_file_path}")

        # Auto-detect scorer name if not provided
        if unique_name is None:
            unique_name = self._extract_scorer_name(scorer_file_path)
            judgeval_logger.info(f"Auto-detected scorer name: '{unique_name}'")

        # Read scorer code
        with open(scorer_file_path, "r") as f:
            scorer_code = f.read()

        # Read requirements (optional)
        requirements_text = ""
        if requirements_file_path and os.path.exists(requirements_file_path):
            with open(requirements_file_path, "r") as f:
                requirements_text = f.read()

        try:
            response = self.api_client.upload_custom_scorer(
                scorer_name=unique_name,
                scorer_code=scorer_code,
                requirements_text=requirements_text,
            )

            if response.get("status") == "success":
                judgeval_logger.info(
                    f"Successfully uploaded custom scorer: {unique_name}"
                )
                return True
            else:
                judgeval_logger.error(f"Failed to upload custom scorer: {unique_name}")
                return False

        except Exception as e:
            judgeval_logger.error(f"Error uploading custom scorer: {e}")
            raise
