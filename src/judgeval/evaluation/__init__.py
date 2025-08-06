from typing import Dict, List, Optional, Union
from pydantic import BaseModel, field_validator, Field

from judgeval.api import JudgmentSyncClient
from judgeval.data import Example
from judgeval.data.result import ScoringResult
from judgeval.data.trace_run import TraceRun
from judgeval.exceptions import JudgmentAPIError
from judgeval.logger import judgeval_logger
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL
from judgeval.constants import ACCEPTABLE_MODELS
from judgeval.data.judgment_types import ScoringResultJudgmentType


class EvaluationRun(BaseModel):
    """
    Stores example and evaluation scorers together for running an eval task

    Args:
        project_name (str): The name of the project the evaluation results belong to
        eval_name (str): A name for this evaluation run
        examples (List[Example]): The examples to evaluate
        scorers (List[Union[JudgmentScorer, BaseScorer]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
        metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
    """

    organization_id: Optional[str] = None
    project_name: Optional[str] = Field(default=None, validate_default=True)
    eval_name: Optional[str] = Field(default=None, validate_default=True)
    examples: List[Example]
    scorers: List[Union[APIScorerConfig, BaseScorer]]
    model: Optional[str] = JUDGMENT_DEFAULT_GPT_MODEL
    trace_span_id: Optional[str] = None
    trace_id: Optional[str] = None
    override: Optional[bool] = False
    append: Optional[bool] = False

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)

        data["scorers"] = [scorer.model_dump() for scorer in self.scorers]
        data["examples"] = [example.model_dump() for example in self.examples]

        return data

    @field_validator("examples")
    def validate_examples(cls, v):
        if not v:
            raise ValueError("Examples cannot be empty.")
        for item in v:
            if not isinstance(item, Example):
                raise ValueError(f"Item of type {type(item)} is not a Example")
        return v

    @field_validator("scorers", mode="before")
    def validate_scorers(cls, v):
        if not v:
            raise ValueError("Scorers cannot be empty.")
        if not all(
            isinstance(scorer, BaseScorer) or isinstance(scorer, APIScorerConfig)
            for scorer in v
        ):
            raise ValueError(
                "All scorers must be of type BaseScorer or APIScorerConfig."
            )
        return v

    @field_validator("model")
    def validate_model(cls, v, values):
        if not v:
            raise ValueError("Model cannot be empty.")

        # Check if model is string or list of strings
        if isinstance(v, str):
            if v not in ACCEPTABLE_MODELS:
                raise ValueError(
                    f"Model name {v} not recognized. Please select a valid model name.)"
                )
            return v

    class Config:
        arbitrary_types_allowed = True


def execute_api_trace_eval(trace_run: TraceRun, judgment_api_key: str) -> Dict:
    # submit API request to execute evals
    if not judgment_api_key or not trace_run.organization_id:
        raise ValueError("API key and organization ID are required")
    api_client = JudgmentSyncClient(judgment_api_key, trace_run.organization_id)
    return api_client.run_trace_evaluation(trace_run.model_dump(warnings=False))


def check_missing_scorer_data(results: List[ScoringResult]) -> List[ScoringResult]:
    """
    Checks if any `ScoringResult` objects are missing `scorers_data`.

    If any are missing, logs an error and returns the results.
    """
    for i, result in enumerate(results):
        if not result.scorers_data:
            judgeval_logger.error(
                f"Scorer data is missing for example {i}. "
                "This is usually caused when the example does not contain "
                "the fields required by the scorer. "
                "Check that your example contains the fields required by the scorers. "
                # TODO: add docs link here for reference.
                "See https://docs.judgmentlabs.ai/sdk-reference/judgment-client#override for more information."
            )
    return results


def check_example_keys(
    keys: List[str],
    eval_name: str,
    project_name: str,
    judgment_api_key: str,
    organization_id: str,
) -> None:
    """
    Checks if the current experiment (if one exists) has the same keys for example
    """
    api_client = JudgmentSyncClient(judgment_api_key, organization_id)
    api_client.check_example_keys(keys, eval_name, project_name)


def check_experiment_type(
    eval_name: str,
    project_name: str,
    judgment_api_key: str,
    organization_id: str,
    is_trace: bool,
) -> None:
    """
    Checks if the current experiment, if one exists, has the same type (examples of traces)
    """
    api_client = JudgmentSyncClient(judgment_api_key, organization_id)
    api_client.check_experiment_type(eval_name, project_name, is_trace)


def check_eval_run_name_exists(
    eval_name: str, project_name: str, judgment_api_key: str, organization_id: str
) -> None:
    """
    Checks if an evaluation run name already exists for a given project.

    Args:
        eval_name (str): Name of the evaluation run
        project_name (str): Name of the project
        judgment_api_key (str): API key for authentication

    Raises:
        ValueError: If the evaluation run name already exists
        JudgmentAPIError: If there's an API error during the check
    """
    api_client = JudgmentSyncClient(judgment_api_key, organization_id)
    api_client.check_eval_run_name_exists(eval_name, project_name)


def log_evaluation_results(
    scoring_results: List[ScoringResult],
    run: Union[EvaluationRun, TraceRun],
    judgment_api_key: str,
) -> str:
    """
    Logs evaluation results to the Judgment API database.

    Args:
        merged_results (List[ScoringResult]): The results to log
        evaluation_run (EvaluationRun): The evaluation run containing project info and API key
        judgment_api_key (str): The API key for the Judgment API

    Raises:
        JudgmentAPIError: If there's an API error during logging
        ValueError: If there's a validation error with the results
    """
    if not judgment_api_key or not run.organization_id:
        raise ValueError("API key and organization ID are required")

    api_client = JudgmentSyncClient(judgment_api_key, run.organization_id)
    response = api_client.log_evaluation_results(
        [result.model_dump(warnings=False) for result in scoring_results],
        run.model_dump(warnings=False),
    )
    url = response.get("ui_results_url")
    return url


def run_evaluation(eval: EvaluationRun) -> List[ScoringResult]:
    keys = eval.examples[0].get_fields().keys()
    for example in eval.examples:
        if example.get_fields().keys() != keys:
            raise ValueError("All examples must have the same fields")

    return []


__all__ = ("EvaluationRun", "run_evaluation")
