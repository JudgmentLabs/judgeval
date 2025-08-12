import asyncio
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, field_validator, Field

from judgeval.api import JudgmentSyncClient
from judgeval.data import Example, ScoringResult
from judgeval.data.trace_run import TraceRun
from judgeval.logger import judgeval_logger
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL, JUDGMENT_MAX_CONCURRENT_EVALUATIONS
from judgeval.constants import ACCEPTABLE_MODELS
from judgeval.exceptions import JudgmentAPIError


class EvaluationRun(BaseModel):
    """
    Stores example and evaluation scorers together for running an eval task
    """

    organization_id: Optional[str] = None
    project_name: Optional[str] = Field(default=None, validate_default=True)
    eval_name: Optional[str] = Field(default=None, validate_default=True)
    examples: List[Example]
    custom_scorers: List[BaseScorer] = Field(default_factory=list)
    judgment_scorers: List[APIScorerConfig] = Field(default_factory=list)
    model: str = JUDGMENT_DEFAULT_GPT_MODEL
    trace_span_id: Optional[str] = None
    trace_id: Optional[str] = None
    override: Optional[bool] = False
    append: Optional[bool] = False

    def __init__(
        self,
        scorers: Optional[List[Union[BaseScorer, APIScorerConfig]]] = None,
        **kwargs,
    ):
        """Initialize EvaluationRun with automatic scorer classification."""
        if scorers is not None:
            custom_scorers = [s for s in scorers if isinstance(s, BaseScorer)]
            judgment_scorers = [s for s in scorers if isinstance(s, APIScorerConfig)]
            kwargs["custom_scorers"] = custom_scorers
            kwargs["judgment_scorers"] = judgment_scorers
        super().__init__(**kwargs)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["custom_scorers"] = [s.model_dump() for s in self.custom_scorers]
        data["judgment_scorers"] = [s.model_dump() for s in self.judgment_scorers]
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

    @field_validator("model")
    def validate_model(cls, v, values):
        if not v:
            raise ValueError("Model cannot be empty.")

        if isinstance(v, str):
            if v not in ACCEPTABLE_MODELS:
                raise ValueError(
                    f"Model name {v} not recognized. Please select a valid model name."
                )
            return v

    class Config:
        arbitrary_types_allowed = True


async def a_execute_scoring(
    examples: List[Example],
    scorers: List[BaseScorer],
    model: str,
    throttle_value: int = 0,
    max_concurrent: int = 10,
) -> List[ScoringResult]:
    """
    Execute scoring locally using custom scorers
    """
    from judgeval.data import generate_scoring_result, create_scorer_data
    import time

    semaphore = asyncio.Semaphore(max_concurrent)

    async def safe_a_score_example(scorer: BaseScorer, example: Example):
        """Safely scores an Example using a BaseScorer"""
        try:
            if hasattr(scorer, "a_score_example"):
                score = await getattr(scorer, "a_score_example")(example)
                if score is None:
                    raise Exception("a_score_example need to return a score")
                elif score < 0:
                    judgeval_logger.warning("score cannot be less than 0, setting to 0")
                    score = 0
                elif score > 1:
                    judgeval_logger.warning(
                        "score cannot be greater than 1, setting to 1"
                    )
                    score = 1
                scorer.score = score
                scorer.success = getattr(scorer, "success_check", lambda: True)()
            else:
                raise Exception(
                    f"Scorer {scorer.score_type} does not have a_score_example method"
                )
        except Exception as e:
            judgeval_logger.error(f"Error during scoring: {str(e)}")
            scorer.error = str(e)
            scorer.success = False
            scorer.score = 0

    async def execute_with_semaphore(func, *args, **kwargs):
        async with semaphore:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                judgeval_logger.error(f"Error executing function: {e}")
                return None

    async def a_eval_examples_helper(
        scorers_list: List[BaseScorer],
        example: Example,
        scoring_results: List[Optional[ScoringResult]],
        score_index: int,
    ):
        """Evaluate a single example using a list of scorers"""
        scoring_start_time = time.perf_counter()

        # Add model to scorers if not present
        for scorer in scorers_list:
            if not getattr(scorer, "model", None):
                scorer.model = model

        # Score the example with all scorers
        tasks = [safe_a_score_example(scorer, example) for scorer in scorers_list]
        await asyncio.gather(*tasks)

        # Collect results
        success = True
        scorer_data_list = []
        for scorer in scorers_list:
            if getattr(scorer, "skipped", False):
                continue
            try:
                scorer_data = create_scorer_data(scorer)
                for s in scorer_data:
                    success = success and s.success
                scorer_data_list.extend(scorer_data)
            except Exception as e:
                judgeval_logger.error(f"Error creating scorer data: {e}")
                success = False

        scoring_end_time = time.perf_counter()
        run_duration = scoring_end_time - scoring_start_time

        scoring_result = generate_scoring_result(
            example, scorer_data_list, run_duration, success
        )
        scoring_results[score_index] = scoring_result

    # Initialize results list
    scoring_results: List[Optional[ScoringResult]] = [None for _ in examples]
    tasks = []

    # Create tasks for each example
    for i, example in enumerate(examples):
        if isinstance(example, Example):
            if len(scorers) == 0:
                continue

            # Clone scorers for each example to avoid conflicts
            cloned_scorers = [scorer.model_copy() for scorer in scorers]

            task = execute_with_semaphore(
                a_eval_examples_helper, cloned_scorers, example, scoring_results, i
            )
            tasks.append(asyncio.create_task(task))

            if throttle_value > 0:
                await asyncio.sleep(throttle_value)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Filter out None results
    return [result for result in scoring_results if result is not None]


def check_missing_scorer_data(results: List[ScoringResult]) -> List[ScoringResult]:
    """
    Checks if any `ScoringResult` objects are missing `scorers_data`.
    """
    for i, result in enumerate(results):
        if not result.scorers_data:
            judgeval_logger.error(
                f"Scorer data is missing for example {i}. "
                "This is usually caused when the example does not contain "
                "the fields required by the scorer."
            )
    return results


def log_evaluation_results(
    scoring_results: List[ScoringResult],
    run: Union[EvaluationRun, TraceRun],
    judgment_api_key: str,
) -> str:
    """
    Logs evaluation results to the Judgment API database.
    """
    try:
        if not judgment_api_key or not run.organization_id:
            raise ValueError("API key and organization ID are required")

        api_client = JudgmentSyncClient(judgment_api_key, run.organization_id)

        # Convert to the format expected by the API
        eval_results: Any = {
            "results": [
                result.model_dump(warnings=False) for result in scoring_results
            ],
            "run": run.model_dump(warnings=False),
        }

        response = api_client.log_eval_results(eval_results)
        return response.get("ui_results_url", "")

    except Exception as e:
        judgeval_logger.error(f"Failed to save evaluation results to DB: {str(e)}")
        raise ValueError(
            f"Request failed while saving evaluation results to DB: {str(e)}"
        )


def execute_api_eval(evaluation_run: EvaluationRun, judgment_api_key: str) -> Dict:
    """
    Executes an evaluation using the Judgment API.
    """
    try:
        if not judgment_api_key or not evaluation_run.organization_id:
            raise ValueError("API key and organization ID are required")

        api_client = JudgmentSyncClient(
            judgment_api_key, evaluation_run.organization_id
        )

        # Convert to API format
        eval_request: Any = {
            "examples": [ex.model_dump() for ex in evaluation_run.examples],
            "model": evaluation_run.model,
        }

        return api_client.evaluate(eval_request)

    except Exception as e:
        judgeval_logger.error(f"Error: {e}")
        raise ValueError(
            f"An error occurred while executing the Judgment API request: {str(e)}"
        )


def run_evaluation(
    evaluation_run: EvaluationRun, judgment_api_key: str
) -> List[ScoringResult]:
    """
    Executes an evaluation of `Example`s using one or more `Scorer`s
    """
    # Validate required fields
    if not evaluation_run.organization_id:
        raise ValueError("organization_id is required")

    # Check that every example has the same keys
    keys = evaluation_run.examples[0].model_dump().keys()
    for example in evaluation_run.examples:
        current_keys = example.model_dump().keys()
        if current_keys != keys:
            raise ValueError(
                f"All examples must have the same keys: {current_keys} != {keys}"
            )

    results: List[ScoringResult] = []
    url = ""

    # Check for mixed scorer types
    if (
        len(evaluation_run.custom_scorers) > 0
        and len(evaluation_run.judgment_scorers) > 0
    ):
        error_msg = "We currently do not support running both local and Judgment API scorers at the same time. Please run your evaluation with either local scorers or Judgment API scorers, but not both."
        judgeval_logger.error(error_msg)
        raise ValueError(error_msg)

    # Check for server-hosted scorers
    e2b_scorers = [
        cs
        for cs in evaluation_run.custom_scorers
        if getattr(cs, "server_hosted", False)
    ]

    if evaluation_run.judgment_scorers or e2b_scorers:
        if evaluation_run.judgment_scorers and e2b_scorers:
            error_msg = "We currently do not support running both hosted custom scorers and Judgment API scorers at the same time. Please run your evaluation with one or the other, but not both."
            judgeval_logger.error(error_msg)
            raise ValueError(error_msg)

        if len(e2b_scorers) > 1:
            error_msg = "We currently do not support running multiple hosted custom scorers at the same time."
            judgeval_logger.error(error_msg)
            raise ValueError(error_msg)

        # Use API-based evaluation
        judgeval_logger.info("Running evaluation using Judgment API scorers...")
        try:
            response_data = execute_api_eval(evaluation_run, judgment_api_key)
            results = [
                ScoringResult(**result) for result in response_data.get("results", [])
            ]
            url = response_data.get("ui_results_url", "")
        except Exception as e:
            raise ValueError(f"Error during API evaluation: {str(e)}")
    else:
        # Use local evaluation - simplified async call
        judgeval_logger.info("Running evaluation using local scorers...")
        results = asyncio.run(
            a_execute_scoring(
                evaluation_run.examples,
                evaluation_run.custom_scorers,
                model=evaluation_run.model,
                throttle_value=0,
                max_concurrent=JUDGMENT_MAX_CONCURRENT_EVALUATIONS,
            )
        )

        # Log results to API
        url = log_evaluation_results(results, evaluation_run, judgment_api_key)

    if url:
        judgeval_logger.info(f"🔍 You can view your evaluation results here: {url}")

    return check_missing_scorer_data(results)


__all__ = ("EvaluationRun", "run_evaluation")
