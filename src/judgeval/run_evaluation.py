from __future__ import annotations

import asyncio
import concurrent.futures
import time
import orjson
import sys
import threading
from typing import List, Dict, Union, Optional, Callable, Tuple, Any, TYPE_CHECKING
from rich import print as rprint

from judgeval.data import ScorerData, ScoringResult, Example, Trace
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.scorers.score import a_execute_scoring
from judgeval.common.api import JudgmentApiClient
from judgeval.constants import (
    MAX_CONCURRENT_EVALUATIONS,
)
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.common.api.api import JudgmentAPIException
from judgeval.common.logger import judgeval_logger


if TYPE_CHECKING:
    from judgeval.common.tracer import Tracer
    from judgeval.data.trace_run import TraceRun
    from judgeval.evaluation_run import EvaluationRun
    from judgeval.integrations.langgraph import JudgevalCallbackHandler


def safe_run_async(coro):
    """
    Safely run an async coroutine whether or not there's already an event loop running.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        # Try to get the running loop
        asyncio.get_running_loop()
        # If we get here, there's already a loop running
        # Run in a separate thread to avoid "asyncio.run() cannot be called from a running event loop"
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No event loop is running, safe to use asyncio.run()
        return asyncio.run(coro)


def send_to_rabbitmq(evaluation_run: EvaluationRun) -> Dict[str, Any]:
    """
    Sends an evaluation run to the RabbitMQ evaluation queue.
    """
    if not evaluation_run.judgment_api_key or not evaluation_run.organization_id:
        raise ValueError("API key and organization ID are required")
    if not evaluation_run.eval_name or not evaluation_run.project_name:
        raise ValueError("Eval name and project name are required")
    api_client = JudgmentApiClient(
        evaluation_run.judgment_api_key, evaluation_run.organization_id
    )
    return api_client.add_to_evaluation_queue(
        evaluation_run.eval_name, evaluation_run.project_name
    )


def execute_api_eval(evaluation_run: EvaluationRun) -> Dict:
    """
    Executes an evaluation of a list of `Example`s using one or more `JudgmentScorer`s via the Judgment API.

    Args:
        evaluation_run (EvaluationRun): The evaluation run object containing the examples, scorers, and metadata

    Returns:
        List[Dict]: The results of the evaluation. Each result is a dictionary containing the fields of a `ScoringResult`
                    object.
    """

    try:
        # submit API request to execute evals
        if not evaluation_run.judgment_api_key or not evaluation_run.organization_id:
            raise ValueError("API key and organization ID are required")
        api_client = JudgmentApiClient(
            evaluation_run.judgment_api_key, evaluation_run.organization_id
        )
        return api_client.run_evaluation(evaluation_run.model_dump())
    except Exception as e:
        judgeval_logger.error(f"Error: {e}")

        details = "No details provided"
        if isinstance(e, JudgmentAPIException):
            details = e.response_json.get("detail", "No details provided")

        raise JudgmentAPIError(
            "An error occurred while executing the Judgment API request: " + details
        )


def execute_api_trace_eval(trace_run: TraceRun, judgment_api_key: str) -> Dict:
    """
    Executes an evaluation of a list of `Trace`s using one or more `JudgmentScorer`s via the Judgment API.
    """

    try:
        # submit API request to execute evals
        if not judgment_api_key or not trace_run.organization_id:
            raise ValueError("API key and organization ID are required")
        api_client = JudgmentApiClient(judgment_api_key, trace_run.organization_id)
        return api_client.run_trace_evaluation(trace_run.model_dump(warnings=False))
    except Exception as e:
        judgeval_logger.error(f"Error: {e}")

        details = "An unknown error occurred."
        if isinstance(e, JudgmentAPIException):
            details = e.response_json.get("detail", "An unknown error occurred.")

        raise JudgmentAPIError(
            "An error occurred while executing the Judgment API request: " + details
        )


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
                "TODO add docs link here for reference."
            )
    return results


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
    api_client = JudgmentApiClient(judgment_api_key, organization_id)

    try:
        api_client.check_experiment_type(eval_name, project_name, is_trace)
    except JudgmentAPIException as e:
        if e.response.status_code == 422:
            judgeval_logger.error(f"{e.response_json}")
            raise ValueError(f"{e.response_json}")
        else:
            raise e
    except Exception as e:
        judgeval_logger.error(f"Failed to check if experiment type exists: {str(e)}")
        raise JudgmentAPIError(f"Failed to check if experiment type exists: {str(e)}")


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
    api_client = JudgmentApiClient(judgment_api_key, organization_id)
    try:
        api_client.check_eval_run_name_exists(eval_name, project_name)
    except JudgmentAPIException as e:
        if e.response.status_code == 409:
            error_str = f"Eval run name '{eval_name}' already exists for this project. Please choose a different name, set the `override` flag to true, or set the `append` flag to true. See https://docs.judgmentlabs.ai/sdk-reference/judgment-client#override for more information."
            judgeval_logger.error(error_str)
            raise ValueError(error_str)
        else:
            raise e

    except Exception as e:
        judgeval_logger.error(f"Failed to check if eval run name exists: {str(e)}")
        raise JudgmentAPIError(f"Failed to check if eval run name exists: {str(e)}")


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
    api_client = JudgmentApiClient(judgment_api_key, organization_id)
    try:
        api_client.check_example_keys(keys, eval_name, project_name)
    except Exception as e:
        judgeval_logger.error(f"Failed to check if example keys match: {str(e)}")
        raise JudgmentAPIError(f"Failed to check if example keys match: {str(e)}")


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
    try:
        if not judgment_api_key or not run.organization_id:
            raise ValueError("API key and organization ID are required")

        api_client = JudgmentApiClient(judgment_api_key, run.organization_id)
        response = api_client.log_evaluation_results(
            scoring_results,
            run.model_dump(warnings=False),
        )
        url = response.get("ui_results_url")
        return url

    except Exception as e:
        judgeval_logger.error(f"Failed to save evaluation results to DB: {str(e)}")
        raise JudgmentAPIError(
            f"Request failed while saving evaluation results to DB: {str(e)}"
        )


def check_examples(
    examples: List[Example], scorers: List[Union[APIScorerConfig, BaseScorer]]
) -> None:
    """
    Checks if the example contains the necessary parameters for the scorer.
    """
    prompt_user = False
    for scorer in scorers:
        for example in examples:
            missing_params = []
            for param in scorer.required_params:
                if getattr(example, param.value) is None:
                    missing_params.append(f"{param.value}")
            if missing_params:
                rprint(
                    f"[yellow]⚠️  WARNING:[/yellow] Example is missing required parameters for scorer [bold]{scorer.score_type.value}[/bold]"
                )
                rprint(f"Missing parameters: {', '.join(missing_params)}")
                rprint(
                    f"Example: {orjson.dumps(example.model_dump(), option=orjson.OPT_INDENT_2).decode('utf-8')}"
                )
                rprint("-" * 40)
                prompt_user = True

    if prompt_user:
        user_input = input("Do you want to continue? (y/n)")
        if user_input.lower() != "y":
            sys.exit(0)
        else:
            rprint("[green]Continuing...[/green]")


def run_trace_eval(
    trace_run: TraceRun,
    judgment_api_key: str,
    override: bool = False,
    function: Optional[Callable] = None,
    tracer: Optional[Union[Tracer, "JudgevalCallbackHandler"]] = None,
    examples: Optional[List[Example]] = None,
) -> List[ScoringResult]:
    # Call endpoint to check to see if eval run name exists (if we DON'T want to override and DO want to log results)
    if not override and not trace_run.append:
        check_eval_run_name_exists(
            trace_run.eval_name,
            trace_run.project_name,
            judgment_api_key,
            trace_run.organization_id,
        )

    if trace_run.append:
        # Check that the current experiment, if one exists, has the same type (examples or traces)
        check_experiment_type(
            trace_run.eval_name,
            trace_run.project_name,
            judgment_api_key,
            trace_run.organization_id,
            True,
        )
    if function and tracer and examples is not None:
        new_traces: List[Trace] = []

        # Handle case where tracer is actually a callback handler
        actual_tracer = tracer
        if hasattr(tracer, "tracer") and hasattr(tracer.tracer, "traces"):
            # This is a callback handler, get the underlying tracer
            actual_tracer = tracer.tracer

        if trace_run.project_name != actual_tracer.project_name:
            raise ValueError(
                f"Project name mismatch between run_trace_eval and tracer. "
                f"Trace run: {trace_run.project_name}, "
                f"Tracer: {actual_tracer.project_name}"
            )

        actual_tracer.offline_mode = True
        actual_tracer.traces = []
        judgeval_logger.info("Running agent function: ")
        for example in examples:
            if example.input:
                if isinstance(example.input, str):
                    function(example.input)
                elif isinstance(example.input, dict):
                    function(**example.input)
                else:
                    raise ValueError(
                        f"Input must be string or dict, got {type(example.input)}"
                    )
            else:
                function()

        for i, trace in enumerate(actual_tracer.traces):
            # We set the root-level trace span with the expected tools of the Trace
            trace = Trace(**trace)
            trace.trace_spans[0].expected_tools = examples[i].expected_tools
            new_traces.append(trace)
        trace_run.traces = new_traces
        actual_tracer.traces = []

    # Execute evaluation using Judgment API
    try:  # execute an EvaluationRun with just JudgmentScorers
        judgeval_logger.info("Executing Trace Evaluation... ")
        response_data: Dict = execute_api_trace_eval(trace_run, judgment_api_key)
        scoring_results = [
            ScoringResult(**result) for result in response_data["results"]
        ]
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            f"An error occurred while executing the Judgment API request: {str(e)}"
        )
    except ValueError as e:
        raise ValueError(
            f"Please check your TraceRun object, one or more fields are invalid: {str(e)}"
        )

    # Convert the response data to `ScoringResult` objects
    # TODO: allow for custom scorer on traces

    url = log_evaluation_results(
        response_data["agent_results"], trace_run, judgment_api_key
    )
    rprint(
        f"\n🔍 You can view your evaluation results here: [rgb(106,0,255)][link={url}]View Results[/link]\n"
    )
    return scoring_results


async def get_evaluation_status(
    eval_name: str, project_name: str, judgment_api_key: str, organization_id: str
) -> Dict:
    """
    Gets the status of an async evaluation run.

    Args:
        eval_name (str): Name of the evaluation run
        project_name (str): Name of the project
        judgment_api_key (str): API key for authentication
        organization_id (str): Organization ID for the evaluation

    Returns:
        Dict: Status information including:
            - status: 'pending', 'running', 'completed', or 'failed'
            - results: List of ScoringResult objects if completed
            - error: Error message if failed
    """
    api_client = JudgmentApiClient(judgment_api_key, organization_id)
    try:
        return api_client.get_evaluation_status(eval_name, project_name)
    except Exception as e:
        raise JudgmentAPIError(
            f"An error occurred while checking evaluation status: {str(e)}"
        )


def retrieve_counts(result: Dict):
    scorer_data_count = 0
    for example in result.get("examples", []):
        for scorer in example.get("scorer_data", []):
            scorer_data_count += 1
    return scorer_data_count


def _poll_evaluation_until_complete(
    eval_name: str,
    project_name: str,
    judgment_api_key: str,
    organization_id: str,
    expected_scorer_data_count: int,
    poll_interval_seconds: float = 5,
    max_failures: int = 5,
    max_poll_count: int = 60,  # This should be equivalent to 5 minutes
) -> Tuple[List[ScoringResult], str]:
    """
    Polls until the evaluation is complete and returns the results.

    Args:
        eval_name (str): Name of the evaluation run
        project_name (str): Name of the project
        judgment_api_key (str): API key for authentication
        organization_id (str): Organization ID for the evaluation
        poll_interval_seconds (int, optional): Time between status checks in seconds. Defaults to 5.
        original_examples (List[Example], optional): The original examples sent for evaluation.
                                                    If provided, will match results with original examples.

    Returns:
        List[ScoringResult]: The evaluation results
    """
    poll_count = 0
    exception_count = 0
    api_client = JudgmentApiClient(judgment_api_key, organization_id)
    while poll_count < max_poll_count:
        poll_count += 1
        try:
            # Check status
            status_response = api_client.get_evaluation_status(eval_name, project_name)

            if status_response.get("status") != "completed":
                time.sleep(poll_interval_seconds)
                continue

            results_response = api_client.fetch_evaluation_results(
                project_name, eval_name
            )
            url = results_response.get("ui_results_url")

            if results_response.get("examples") is None:
                time.sleep(poll_interval_seconds)
                continue

            examples_data = results_response.get("examples", [])
            scoring_results = []
            scorer_data_count = 0

            for example_data in examples_data:
                scorer_data_list = []
                for raw_scorer_data in example_data.get("scorer_data", []):
                    scorer_data = ScorerData(**raw_scorer_data)
                    scorer_data_list.append(scorer_data)
                    scorer_data_count += 1

                example = Example(**example_data)

                success = all(scorer_data.success for scorer_data in scorer_data_list)
                scoring_result = ScoringResult(
                    success=success,
                    scorers_data=scorer_data_list,
                    data_object=example,
                )
                scoring_results.append(scoring_result)

            if scorer_data_count != expected_scorer_data_count:
                time.sleep(poll_interval_seconds)
                continue

            return scoring_results, url
        except Exception as e:
            exception_count += 1
            if isinstance(e, JudgmentAPIError):
                raise

            judgeval_logger.error(f"Error checking evaluation status: {str(e)}")
            if exception_count > max_failures:
                raise JudgmentAPIError(
                    f"Error checking evaluation status after {poll_count} attempts: {str(e)}"
                )

            time.sleep(poll_interval_seconds)

    raise JudgmentAPIError(
        f"Error checking evaluation status after {poll_count} attempts"
    )


def progress_logger(stop_event, msg="Working...", interval=5):
    start = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - start)
        judgeval_logger.info(f"{msg} ({elapsed} sec)")
        stop_event.wait(interval)


def run_eval(
    evaluation_run: EvaluationRun,
    judgment_api_key: str,
    override: bool = False,
) -> List[ScoringResult]:
    """
    Executes an evaluation of `Example`s using one or more `Scorer`s

    Args:
        evaluation_run (EvaluationRun): Stores example and evaluation together for running
        override (bool, optional): Whether to override existing evaluation run with same name. Defaults to False.

    Returns:
        List[ScoringResult]: A list of ScoringResult objects
    """
    # Check that every example has the same keys
    keys = evaluation_run.examples[0].get_fields().keys()
    for example in evaluation_run.examples:
        current_keys = example.get_fields().keys()
        if current_keys != keys:
            raise ValueError(
                f"All examples must have the same keys: {current_keys} != {keys}"
            )

    # Call endpoint to check to see if eval run name exists (if we DON'T want to override and DO want to log results)
    if not override and not evaluation_run.append:
        check_eval_run_name_exists(
            evaluation_run.eval_name,
            evaluation_run.project_name,
            judgment_api_key,
            evaluation_run.organization_id,
        )

    if evaluation_run.append:
        # Check that the current experiment, if one exists, has the same type (examples of traces)
        check_experiment_type(
            evaluation_run.eval_name,
            evaluation_run.project_name,
            judgment_api_key,
            evaluation_run.organization_id,
            False,
        )

        # Ensure that current experiment (if one exists) has the same keys for example
        check_example_keys(
            keys=list(keys),
            eval_name=evaluation_run.eval_name,
            project_name=evaluation_run.project_name,
            judgment_api_key=judgment_api_key,
            organization_id=evaluation_run.organization_id,
        )

    judgment_scorers: List[APIScorerConfig] = []
    local_scorers: List[BaseScorer] = []
    for scorer in evaluation_run.scorers:
        if isinstance(scorer, APIScorerConfig):
            judgment_scorers.append(scorer)
        else:
            local_scorers.append(scorer)

    results: List[ScoringResult] = []
    url = ""

    if len(local_scorers) > 0 and len(judgment_scorers) > 0:
        error_msg = "We currently do not support running both local and Judgment API scorers at the same time. Please run your evaluation with either local scorers or Judgment API scorers, but not both."
        judgeval_logger.error(error_msg)
        raise ValueError(error_msg)

    if len(judgment_scorers) > 0:
        check_examples(evaluation_run.examples, judgment_scorers)
        stop_event = threading.Event()
        t = threading.Thread(
            target=progress_logger, args=(stop_event, "Running evaluation...")
        )
        t.start()
        try:
            api_client = JudgmentApiClient(
                judgment_api_key, evaluation_run.organization_id
            )
            response = api_client.add_to_evaluation_queue(
                evaluation_run.model_dump(warnings=False)
            )

            if not response.get("success", False):
                error_message = response.error
                judgeval_logger.error(
                    f"Error adding evaluation to queue: {error_message}"
                )
                raise JudgmentAPIError(error_message)

            old_scorer_data_count = 0
            if evaluation_run.append:
                try:
                    results_response = api_client.fetch_evaluation_results(
                        evaluation_run.project_name, evaluation_run.eval_name
                    )
                    old_scorer_data_count = retrieve_counts(results_response)
                except Exception:
                    # This usually means the user did append = True but the eval run name doesn't exist yet
                    pass

            results, url = _poll_evaluation_until_complete(
                eval_name=evaluation_run.eval_name,
                project_name=evaluation_run.project_name,
                judgment_api_key=judgment_api_key,
                organization_id=evaluation_run.organization_id,
                expected_scorer_data_count=(
                    len(evaluation_run.scorers) * len(evaluation_run.examples)
                )
                + old_scorer_data_count,
            )
        finally:
            stop_event.set()
            t.join()

    if len(local_scorers) > 0:
        results = safe_run_async(
            a_execute_scoring(
                evaluation_run.examples,
                local_scorers,
                model=evaluation_run.model,
                throttle_value=0,
                max_concurrent=MAX_CONCURRENT_EVALUATIONS,
            )
        )

        send_results = [
            scoring_result.model_dump(warnings=False) for scoring_result in results
        ]
        url = log_evaluation_results(send_results, evaluation_run, judgment_api_key)
    rprint(
        f"\n🔍 You can view your evaluation results here: [rgb(106,0,255)][link={url}]View Results[/link]\n"
    )
    return results


def assert_test(scoring_results: List[ScoringResult]) -> None:
    """
    Collects all failed scorers from the scoring results.

    Args:
        ScoringResults (List[ScoringResult]): List of scoring results to check

    Returns:
        None. Raises exceptions for any failed test cases.
    """
    failed_cases: List[ScorerData] = []

    for result in scoring_results:
        if not result.success:
            # Create a test case context with all relevant fields
            test_case: Dict = {"failed_scorers": []}
            if result.scorers_data:
                # If the result was not successful, check each scorer_data
                for scorer_data in result.scorers_data:
                    if not scorer_data.success:
                        if scorer_data.name == "Tool Order":
                            # Remove threshold, evaluation model for Tool Order scorer
                            scorer_data.threshold = None
                            scorer_data.evaluation_model = None
                        test_case["failed_scorers"].append(scorer_data)
            failed_cases.append(test_case)

    if failed_cases:
        error_msg = "The following test cases failed: \n"
        for fail_case in failed_cases:
            for fail_scorer in fail_case["failed_scorers"]:
                error_msg += (
                    f"\nScorer Name: {fail_scorer.name}\n"
                    f"Threshold: {fail_scorer.threshold}\n"
                    f"Success: {fail_scorer.success}\n"
                    f"Score: {fail_scorer.score}\n"
                    f"Reason: {fail_scorer.reason}\n"
                    f"Strict Mode: {fail_scorer.strict_mode}\n"
                    f"Evaluation Model: {fail_scorer.evaluation_model}\n"
                    f"Error: {fail_scorer.error}\n"
                    f"Additional Metadata: {fail_scorer.additional_metadata}\n"
                )
            error_msg += "-" * 100

        total_tests = len(scoring_results)
        failed_tests = len(failed_cases)
        passed_tests = total_tests - failed_tests

        # Print summary with colors
        rprint("\n" + "=" * 80)
        if failed_tests == 0:
            rprint(
                f"[bold green]🎉 ALL TESTS PASSED! {passed_tests}/{total_tests} tests successful[/bold green]"
            )
        else:
            rprint(
                f"[bold red]⚠️  TEST RESULTS: {passed_tests}/{total_tests} passed ({failed_tests} failed)[/bold red]"
            )
        rprint("=" * 80 + "\n")

        # Print individual test cases
        for i, result in enumerate(scoring_results):
            test_num = i + 1
            if result.success:
                rprint(f"[green]✓ Test {test_num}: PASSED[/green]")
            else:
                rprint(f"[red]✗ Test {test_num}: FAILED[/red]")
                if result.scorers_data:
                    for scorer_data in result.scorers_data:
                        if not scorer_data.success:
                            rprint(f"  [yellow]Scorer: {scorer_data.name}[/yellow]")
                            rprint(f"  [red]  Score: {scorer_data.score}[/red]")
                            rprint(f"  [red]  Reason: {scorer_data.reason}[/red]")
                            if scorer_data.error:
                                rprint(f"  [red]  Error: {scorer_data.error}[/red]")
                rprint("  " + "-" * 40)

        rprint("\n" + "=" * 80)
        if failed_tests > 0:
            raise AssertionError(failed_cases)
