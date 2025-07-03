import asyncio
import concurrent.futures
from requests import exceptions
from judgeval.utils.requests import requests
import time
import json
import sys
import itertools
import threading
from typing import List, Dict, Any, Union, Optional, Callable
from rich import print as rprint

from judgeval.data import ScorerData, ScoringResult, Example, Trace
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.scorers.score import a_execute_scoring
from judgeval.constants import (
    ROOT_API,
    JUDGMENT_EVAL_API_URL,
    JUDGMENT_TRACE_EVAL_API_URL,
    JUDGMENT_EVAL_LOG_API_URL,
    MAX_CONCURRENT_EVALUATIONS,
    JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL,
    JUDGMENT_GET_EVAL_STATUS_API_URL,
    JUDGMENT_EVAL_FETCH_API_URL,
)
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.common.logger import debug, info, error, warning, example_logging_context
from judgeval.evaluation_run import EvaluationRun
from judgeval.data.trace_run import TraceRun
from judgeval.common.tracer import Tracer
from langchain_core.callbacks import BaseCallbackHandler


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


def send_to_rabbitmq(evaluation_run: EvaluationRun) -> None:
    """
    Sends an evaluation run to the RabbitMQ evaluation queue.
    """
    payload = evaluation_run.model_dump(warnings=False)
    response = requests.post(
        JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {evaluation_run.judgment_api_key}",
            "X-Organization-Id": evaluation_run.organization_id,
        },
        json=payload,
        verify=True,
    )
    return response.json()


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
        payload = evaluation_run.model_dump()
        response = requests.post(
            JUDGMENT_EVAL_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {evaluation_run.judgment_api_key}",
                "X-Organization-Id": evaluation_run.organization_id,
            },
            json=payload,
            verify=True,
        )
        response_data = response.json()
    except Exception as e:
        error(f"Error: {e}")
        details = response.json().get("detail", "No details provided")
        raise JudgmentAPIError(
            "An error occurred while executing the Judgment API request: " + details
        )
    # Check if the response status code is not 2XX
    # Add check for the duplicate eval run name
    if not response.ok:
        error_message = response_data.get("detail", "An unknown error occurred.")
        error(f"Error: {error_message=}")
        raise JudgmentAPIError(error_message)
    return response_data


def execute_api_trace_eval(trace_run: TraceRun) -> Dict:
    """
    Executes an evaluation of a list of `Trace`s using one or more `JudgmentScorer`s via the Judgment API.
    """

    try:
        # submit API request to execute evals
        payload = trace_run.model_dump(warnings=False)
        response = requests.post(
            JUDGMENT_TRACE_EVAL_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {trace_run.judgment_api_key}",
                "X-Organization-Id": trace_run.organization_id,
            },
            json=payload,
            verify=True,
        )
        response_data = response.json()
    except Exception as e:
        error(f"Error: {e}")
        details = response.json().get("detail", "No details provided")
        raise JudgmentAPIError(
            "An error occurred while executing the Judgment API request: " + details
        )
    # Check if the response status code is not 2XX
    # Add check for the duplicate eval run name
    if not response.ok:
        error_message = response_data.get("detail", "An unknown error occurred.")
        error(f"Error: {error_message=}")
        raise JudgmentAPIError(error_message)
    return response_data


def merge_results(
    api_results: List[ScoringResult], local_results: List[ScoringResult]
) -> List[ScoringResult]:
    """
    When executing scorers that come from both the Judgment API and local scorers, we're left with
    results for each type of scorer. This function merges the results from the API and local evaluations,
    grouped by example. In particular, we merge the `scorers_data` field of each `ScoringResult` object.

    Args:
        api_results (List[ScoringResult]): The `ScoringResult`s from the API evaluation
        local_results (List[ScoringResult]): The `ScoringResult`s from the local evaluation

    Returns:
        List[ScoringResult]: The merged `ScoringResult`s (updated `scorers_data` field)
    """
    # No merge required
    if not local_results and api_results:
        return [result.model_copy() for result in api_results]
    if not api_results and local_results:
        return [result.model_copy() for result in local_results]

    if len(api_results) != len(local_results):
        # Results should be of same length because each ScoringResult is a 1-1 mapping to an Example
        raise ValueError(
            f"The number of API and local results do not match: {len(api_results)} vs {len(local_results)}"
        )

    # Create a copy of api_results to avoid modifying the input
    merged_results = [result.model_copy() for result in api_results]

    # Each ScoringResult in api and local have all the same fields besides `scorers_data`
    for merged_result, local_result in zip(merged_results, local_results):
        if not (merged_result.data_object and local_result.data_object):
            raise ValueError("Data object is None in one of the results.")
        if merged_result.data_object.input != local_result.data_object.input:
            raise ValueError("The API and local results are not aligned.")
        if (
            merged_result.data_object.actual_output
            != local_result.data_object.actual_output
        ):
            raise ValueError("The API and local results are not aligned.")
        if (
            merged_result.data_object.expected_output
            != local_result.data_object.expected_output
        ):
            raise ValueError("The API and local results are not aligned.")
        if merged_result.data_object.context != local_result.data_object.context:
            raise ValueError("The API and local results are not aligned.")
        if (
            merged_result.data_object.retrieval_context
            != local_result.data_object.retrieval_context
        ):
            raise ValueError("The API and local results are not aligned.")
        if (
            merged_result.data_object.additional_metadata
            != local_result.data_object.additional_metadata
        ):
            raise ValueError("The API and local results are not aligned.")
        if (
            merged_result.data_object.tools_called
            != local_result.data_object.tools_called
        ):
            raise ValueError("The API and local results are not aligned.")
        if (
            merged_result.data_object.expected_tools
            != local_result.data_object.expected_tools
        ):
            raise ValueError("The API and local results are not aligned.")

        # Merge ScorerData from the API and local scorers together
        api_scorer_data = merged_result.scorers_data
        local_scorer_data = local_result.scorers_data
        if api_scorer_data is None and local_scorer_data is not None:
            merged_result.scorers_data = local_scorer_data
        elif api_scorer_data is not None and local_scorer_data is not None:
            merged_result.scorers_data = api_scorer_data + local_scorer_data

    return merged_results


def check_missing_scorer_data(results: List[ScoringResult]) -> List[ScoringResult]:
    """
    Checks if any `ScoringResult` objects are missing `scorers_data`.

    If any are missing, logs an error and returns the results.
    """
    for i, result in enumerate(results):
        if not result.scorers_data:
            error(
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
    try:
        response = requests.post(
            f"{ROOT_API}/check_experiment_type/",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {judgment_api_key}",
                "X-Organization-Id": organization_id,
            },
            json={
                "eval_name": eval_name,
                "project_name": project_name,
                "judgment_api_key": judgment_api_key,
                "is_trace": is_trace,
            },
            verify=True,
        )

        if response.status_code == 422:
            error(f"{response.json()}")
            raise ValueError(f"{response.json()}")

        if not response.ok:
            response_data = response.json()
            error_message = response_data.get("detail", "An unknown error occurred.")
            error(f"Error checking eval run name: {error_message}")
            raise JudgmentAPIError(error_message)

    except exceptions.RequestException as e:
        error(f"Failed to check if experiment type exists: {str(e)}")
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
    try:
        response = requests.post(
            f"{ROOT_API}/eval-run-name-exists/",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {judgment_api_key}",
                "X-Organization-Id": organization_id,
            },
            json={
                "eval_name": eval_name,
                "project_name": project_name,
                "judgment_api_key": judgment_api_key,
            },
            verify=True,
        )

        if response.status_code == 409:
            error(
                f"Eval run name '{eval_name}' already exists for this project. Please choose a different name, set the `override` flag to true, or set the `append` flag to true."
            )
            raise ValueError(
                f"Eval run name '{eval_name}' already exists for this project. Please choose a different name, set the `override` flag to true, or set the `append` flag to true."
            )

        if not response.ok:
            response_data = response.json()
            error_message = response_data.get("detail", "An unknown error occurred.")
            error(f"Error checking eval run name: {error_message}")
            raise JudgmentAPIError(error_message)

    except exceptions.RequestException as e:
        error(f"Failed to check if eval run name exists: {str(e)}")
        raise JudgmentAPIError(f"Failed to check if eval run name exists: {str(e)}")


def log_evaluation_results(
    scoring_results: List[ScoringResult], run: Union[EvaluationRun, TraceRun]
) -> str | None:
    """
    Logs evaluation results to the Judgment API database.

    Args:
        merged_results (List[ScoringResult]): The results to log
        evaluation_run (EvaluationRun): The evaluation run containing project info and API key

    Raises:
        JudgmentAPIError: If there's an API error during logging
        ValueError: If there's a validation error with the results
    """
    try:
        res = requests.post(
            JUDGMENT_EVAL_LOG_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {run.judgment_api_key}",
                "X-Organization-Id": run.organization_id,
            },
            json={"results": scoring_results, "run": run.model_dump(warnings=False)},
            verify=True,
        )

        if not res.ok:
            response_data = res.json()
            error_message = response_data.get("detail", "An unknown error occurred.")
            error(f"Error {res.status_code}: {error_message}")
            raise JudgmentAPIError(error_message)

        if "ui_results_url" in res.json():
            url = res.json()["ui_results_url"]
            pretty_str = f"\n🔍 You can view your evaluation results here: [rgb(106,0,255)][link={url}]View Results[/link]\n"
            return pretty_str

        return None

    except exceptions.RequestException as e:
        error(f"Request failed while saving evaluation results to DB: {str(e)}")
        raise JudgmentAPIError(
            f"Request failed while saving evaluation results to DB: {str(e)}"
        )
    except Exception as e:
        error(f"Failed to save evaluation results to DB: {str(e)}")
        raise ValueError(f"Failed to save evaluation results to DB: {str(e)}")


def run_with_spinner(message: str, func, *args, **kwargs) -> Any:
    """Run a function with a spinner in the terminal."""
    spinner = itertools.cycle(["|", "/", "-", "\\"])

    def display_spinner():
        while not stop_spinner_event.is_set():
            sys.stdout.write(f"\r{message}{next(spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)

    stop_spinner_event = threading.Event()
    spinner_thread = threading.Thread(target=display_spinner)
    spinner_thread.start()

    try:
        if asyncio.iscoroutinefunction(func):
            coro = func(*args, **kwargs)
            result = safe_run_async(coro)
        else:
            result = func(*args, **kwargs)
    except Exception as e:
        error(f"An error occurred: {str(e)}")
        stop_spinner_event.set()
        spinner_thread.join()
        raise e
    finally:
        stop_spinner_event.set()
        spinner_thread.join()

        sys.stdout.write("\r" + " " * (len(message) + 1) + "\r")
        sys.stdout.flush()

    return result


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
                rprint(f"Example: {json.dumps(example.model_dump(), indent=2)}")
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
    override: bool = False,
    function: Optional[Callable] = None,
    tracer: Optional[Union[Tracer, BaseCallbackHandler]] = None,
    examples: Optional[List[Example]] = None,
) -> List[ScoringResult]:
    # Call endpoint to check to see if eval run name exists (if we DON'T want to override and DO want to log results)
    if not override and not trace_run.append:
        check_eval_run_name_exists(
            trace_run.eval_name,
            trace_run.project_name,
            trace_run.judgment_api_key,
            trace_run.organization_id,
        )

    if trace_run.append:
        # Check that the current experiment, if one exists, has the same type (examples or traces)
        check_experiment_type(
            trace_run.eval_name,
            trace_run.project_name,
            trace_run.judgment_api_key,
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

        actual_tracer.offline_mode = True
        actual_tracer.traces = []
        for example in examples:
            if example.input:
                if isinstance(example.input, str):
                    run_with_spinner(
                        "Running agent function: ", function, example.input
                    )
                elif isinstance(example.input, dict):
                    run_with_spinner(
                        "Running agent function: ", function, **example.input
                    )
                else:
                    raise ValueError(
                        f"Input must be string or dict, got {type(example.input)}"
                    )
            else:
                run_with_spinner("Running agent function: ", function)

        for i, trace in enumerate(actual_tracer.traces):
            # We set the root-level trace span with the expected tools of the Trace
            trace = Trace(**trace)
            trace.trace_spans[0].expected_tools = examples[i].expected_tools
            new_traces.append(trace)
        trace_run.traces = new_traces
        actual_tracer.traces = []

    # Execute evaluation using Judgment API
    info("Starting API evaluation")
    try:  # execute an EvaluationRun with just JudgmentScorers
        debug("Sending request to Judgment API")
        response_data: Dict = run_with_spinner(
            "Running Trace Evaluation: ", execute_api_trace_eval, trace_run
        )
        scoring_results = [
            ScoringResult(**result) for result in response_data["results"]
        ]
        info(f"Received {len(scoring_results)} results from API")
    except JudgmentAPIError as e:
        error(f"An error occurred while executing the Judgment API request: {str(e)}")
        raise JudgmentAPIError(
            f"An error occurred while executing the Judgment API request: {str(e)}"
        )
    except ValueError as e:
        raise ValueError(
            f"Please check your TraceRun object, one or more fields are invalid: {str(e)}"
        )

    # Convert the response data to `ScoringResult` objects
    debug("Processing API results")
    # TODO: allow for custom scorer on traces

    pretty_str = run_with_spinner(
        "Logging Results: ",
        log_evaluation_results,
        response_data["agent_results"],
        trace_run,
    )
    rprint(pretty_str)

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
    try:
        response = requests.get(
            JUDGMENT_GET_EVAL_STATUS_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {judgment_api_key}",
                "X-Organization-Id": organization_id,
            },
            params={
                "eval_name": eval_name,
                "project_name": project_name,
            },
            verify=True,
        )

        if not response.ok:
            error_message = response.json().get("detail", "An unknown error occurred.")
            error(f"Error checking evaluation status: {error_message}")
            raise JudgmentAPIError(error_message)

        return response.json()
    except exceptions.RequestException as e:
        error(f"Failed to check evaluation status: {str(e)}")
        raise JudgmentAPIError(f"Failed to check evaluation status: {str(e)}")


async def _poll_evaluation_until_complete(
    eval_name: str,
    project_name: str,
    judgment_api_key: str,
    organization_id: str,
    poll_interval_seconds: int = 5,
    original_examples: Optional[List[Example]] = None,
) -> List[ScoringResult]:
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
    # Create example_id to Example mapping if original examples are provided
    original_example_map = {}
    if original_examples:
        for example in original_examples:
            original_example_map[example.example_id] = example

    # Remove the expected scorer names extraction and checking
    # We'll instead verify all examples have consistent scorer data
    while True:
        poll_count += 1
        try:
            # Log polling attempt
            if poll_count % 4 == 0:  # Log every 4th poll to avoid excess logging
                info(
                    f"Polling for evaluation '{eval_name}' in project '{project_name}' (attempt {poll_count})"
                )

            # Check status
            response = await asyncio.to_thread(
                requests.get,
                JUDGMENT_GET_EVAL_STATUS_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {judgment_api_key}",
                    "X-Organization-Id": organization_id,
                },
                params={"eval_name": eval_name, "project_name": project_name},
                verify=True,
            )

            if not response.ok:
                error_message = response.json().get(
                    "detail", "An unknown error occurred."
                )
                error(f"Error checking evaluation status: {error_message}")
                # Don't raise exception immediately, just log and continue polling
                await asyncio.sleep(poll_interval_seconds)
                continue

            status_data = response.json()
            status = status_data.get("status")

            # If complete, get results and return
            if status == "completed" or status == "complete":
                info(
                    f"Evaluation '{eval_name}' reported as completed, fetching and verifying results..."
                )
                results_response = await asyncio.to_thread(
                    requests.post,
                    JUDGMENT_EVAL_FETCH_API_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {judgment_api_key}",
                        "X-Organization-Id": organization_id,
                    },
                    json={"project_name": project_name, "eval_name": eval_name},
                    verify=True,
                )

                if not results_response.ok:
                    error_message = results_response.json().get(
                        "detail", "An unknown error occurred."
                    )
                    error(f"Error fetching evaluation results: {error_message}")
                    raise JudgmentAPIError(error_message)

                result_data = results_response.json()

                if "examples" in result_data:
                    examples_data = result_data.get("examples", [])

                    info(
                        f"Successfully fetched {len(examples_data)} results for evaluation '{eval_name}'"
                    )

                    # Check for result validity if original examples are provided
                    if original_example_map:
                        # Verify all returned examples have matching original examples
                        has_invalid_results = False
                        for example_data in examples_data:
                            example_id = example_data.get("example_id")

                            if example_id not in original_example_map:
                                warning(
                                    f"Server returned example with ID {example_id} not found in original examples. "
                                    + "This indicates stale or incorrect data. Continuing to poll..."
                                )
                                has_invalid_results = True
                                break

                        # If any invalid examples found, continue polling
                        if has_invalid_results:
                            info("Detected stale data. Waiting before polling again...")
                            await asyncio.sleep(poll_interval_seconds)
                            continue

                        # Check if we received the expected number of results
                        if original_examples and len(original_examples) != len(
                            examples_data
                        ):
                            warning(
                                f"Expected {len(original_examples)} results but got {len(examples_data)} results. "
                                + "This indicates incomplete data. Continuing to poll..."
                            )
                            await asyncio.sleep(poll_interval_seconds)
                            continue

                        # Collect all example IDs from scorer data
                        scorer_example_ids = set()
                        for example_data in examples_data:
                            scorer_data_list = example_data.get("scorer_data", [])
                            for scorer_data in scorer_data_list:
                                if "example_id" in scorer_data:
                                    scorer_example_ids.add(scorer_data["example_id"])

                        # Get the set of original example IDs
                        original_example_ids = set(original_example_map.keys())

                        # Check if the sets are equal
                        missing_in_scorer = original_example_ids - scorer_example_ids
                        extra_in_scorer = scorer_example_ids - original_example_ids

                        if missing_in_scorer or extra_in_scorer:
                            if missing_in_scorer:
                                warning(
                                    f"Examples missing in scorer data: {missing_in_scorer}"
                                )
                            if extra_in_scorer:
                                warning(
                                    f"Extra examples in scorer data: {extra_in_scorer}"
                                )
                            info(
                                "Detected mismatched example IDs in scorer data. Waiting before polling again..."
                            )
                            await asyncio.sleep(poll_interval_seconds)
                            continue

                    # Create ScoringResult objects from the raw data
                    scoring_results = []

                    for example_data in examples_data:
                        # Extract example_id from the server response
                        example_id = example_data.get("example_id")

                        # Create ScorerData objects
                        scorer_data_list = []
                        for raw_scorer_data in example_data.get("scorer_data", []):
                            scorer_data_list.append(ScorerData(**raw_scorer_data))

                        # Use the original Example object if we have it and the ID matches
                        if original_example_map:
                            example = original_example_map[example_id]
                            debug(f"Matched result with original example {example_id}")
                        else:
                            # Create Example from example data (excluding scorer_data) if no original examples provided
                            example_dict = {
                                k: v
                                for k, v in example_data.items()
                                if k != "scorer_data"
                            }
                            example = Example(**example_dict)

                        # Calculate success based on whether all scorer_data entries were successful
                        success = (
                            all(scorer_data.success for scorer_data in scorer_data_list)
                            if scorer_data_list
                            else False
                        )

                        # Create ScoringResult
                        scoring_result = ScoringResult(
                            success=success,  # Set based on all scorer data success values
                            scorers_data=scorer_data_list,
                            data_object=example,
                        )
                        scoring_results.append(scoring_result)

                    # If we got here, all validation checks passed
                    info(
                        f"Verified complete results for all {len(scoring_results)} examples with all expected scorer data"
                    )
                    return scoring_results
                else:
                    # No examples found
                    info(
                        f"No example results found for completed evaluation '{eval_name}'"
                    )
                    return []

            elif status == "failed":
                # Evaluation failed
                error_message = status_data.get("error", "Unknown error")
                error(f"Evaluation '{eval_name}' failed: {error_message}")
                raise JudgmentAPIError(f"Evaluation failed: {error_message}")

            elif status == "pending" or status == "running":
                # Only log occasionally for pending/running to avoid flooding logs
                if poll_count % 4 == 0:
                    info(f"Evaluation '{eval_name}' status: {status}")

            # Wait before checking again
            await asyncio.sleep(poll_interval_seconds)

        except Exception as e:
            if isinstance(e, JudgmentAPIError):
                raise

            # For other exceptions, log and continue polling
            error(f"Error checking evaluation status: {str(e)}")
            if poll_count > 20:  # Only raise exception after many failed attempts
                raise JudgmentAPIError(
                    f"Error checking evaluation status after {poll_count} attempts: {str(e)}"
                )

            # Continue polling after a delay
            await asyncio.sleep(poll_interval_seconds)


async def await_with_spinner(task, message: str = "Awaiting async task: "):
    """
    Display a spinner while awaiting an async task.

    Args:
        task: The asyncio task to await
        message (str): Message to display with the spinner

    Returns:
        Any: The result of the awaited task
    """
    spinner = itertools.cycle(["|", "/", "-", "\\"])

    # Create an event to signal when to stop the spinner
    stop_spinner_event = asyncio.Event()

    async def display_spinner():
        while not stop_spinner_event.is_set():
            sys.stdout.write(f"\r{message}{next(spinner)}")
            sys.stdout.flush()
            await asyncio.sleep(0.1)

    # Start the spinner in a separate task
    spinner_task = asyncio.create_task(display_spinner())

    try:
        # Await the actual task
        result = await task
    finally:
        # Signal the spinner to stop and wait for it to finish
        stop_spinner_event.set()
        await spinner_task

        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(message) + 1) + "\r")
        sys.stdout.flush()

    return result


class SpinnerWrappedTask:
    """
    A wrapper for an asyncio task that displays a spinner when awaited.
    """

    def __init__(self, task, message: str):
        self.task = task
        self.message = message

    def __await__(self):
        async def _spin_and_await():
            # self.task resolves to (scoring_results, pretty_str_to_print)
            task_result_tuple = await await_with_spinner(self.task, self.message)

            # Unpack the tuple
            scoring_results, pretty_str_to_print = task_result_tuple

            # Print the pretty string if it exists, after spinner is cleared
            if pretty_str_to_print:
                rprint(pretty_str_to_print)

            # Return only the scoring_results to the original awaiter
            return scoring_results

        return _spin_and_await().__await__()

    # Proxy all Task attributes and methods to the underlying task
    def __getattr__(self, name):
        return getattr(self.task, name)


def run_eval(
    evaluation_run: EvaluationRun,
    override: bool = False,
    async_execution: bool = False,
) -> Union[List[ScoringResult], asyncio.Task, SpinnerWrappedTask]:
    """
    Executes an evaluation of `Example`s using one or more `Scorer`s

    Args:
        evaluation_run (EvaluationRun): Stores example and evaluation together for running
        override (bool, optional): Whether to override existing evaluation run with same name. Defaults to False.
        async_execution (bool, optional): Whether to execute the evaluation asynchronously. Defaults to False.

    Returns:
        Union[List[ScoringResult], Union[asyncio.Task, SpinnerWrappedTask]]:
            - If async_execution is False, returns a list of ScoringResult objects
            - If async_execution is True, returns a Task that will resolve to a list of ScoringResult objects when awaited
    """

    # Call endpoint to check to see if eval run name exists (if we DON'T want to override and DO want to log results)
    if not override and not evaluation_run.append:
        check_eval_run_name_exists(
            evaluation_run.eval_name,
            evaluation_run.project_name,
            evaluation_run.judgment_api_key,
            evaluation_run.organization_id,
        )

    if evaluation_run.append:
        # Check that the current experiment, if one exists, has the same type (examples of traces)
        check_experiment_type(
            evaluation_run.eval_name,
            evaluation_run.project_name,
            evaluation_run.judgment_api_key,
            evaluation_run.organization_id,
            False,
        )

    # Set example IDs if not already set
    debug("Initializing examples with IDs and timestamps")
    for idx, example in enumerate(evaluation_run.examples):
        example.example_index = idx  # Set numeric index
        with example_logging_context(example.created_at, example.example_id):
            debug(
                f"Initialized example {example.example_id} (index: {example.example_index})"
            )
            debug(f"Input: {example.input}")
            debug(f"Actual output: {example.actual_output}")
            if example.expected_output:
                debug(f"Expected output: {example.expected_output}")
            if example.context:
                debug(f"Context: {example.context}")
            if example.retrieval_context:
                debug(f"Retrieval context: {example.retrieval_context}")
            if example.additional_metadata:
                debug(f"Additional metadata: {example.additional_metadata}")
            if example.tools_called:
                debug(f"Tools called: {example.tools_called}")
            if example.expected_tools:
                debug(f"Expected tools: {example.expected_tools}")

    debug(f"Starting evaluation run with {len(evaluation_run.examples)} examples")

    # Group APIScorerConfigs and BaseScorers, then evaluate them in parallel
    debug("Grouping scorers by type")
    judgment_scorers: List[APIScorerConfig] = []
    local_scorers: List[BaseScorer] = []
    for scorer in evaluation_run.scorers:
        if isinstance(scorer, APIScorerConfig):
            judgment_scorers.append(scorer)
            debug(f"Added judgment scorer: {type(scorer).__name__}")
        else:
            local_scorers.append(scorer)
            debug(f"Added local scorer: {type(scorer).__name__}")

    custom_example_check = [scorer.custom_example for scorer in local_scorers]
    if any(custom_example_check) and not all(custom_example_check):
        error("All scorers must be custom scorers if using custom examples")
        raise ValueError("All scorers must be custom scorers if using custom examples")

    debug(
        f"Found {len(judgment_scorers)} judgment scorers and {len(local_scorers)} local scorers"
    )

    api_results: List[ScoringResult] = []
    local_results: List[ScoringResult] = []

    if async_execution:
        if len(local_scorers) > 0:
            error("Local scorers are not supported in async execution")
            raise ValueError("Local scorers are not supported in async execution")

        check_examples(evaluation_run.examples, evaluation_run.scorers)
        info("Starting async evaluation")

        async def _async_evaluation_workflow():
            # Create a payload
            payload = evaluation_run.model_dump(warnings=False)

            # Send the evaluation to the queue
            response = await asyncio.to_thread(
                requests.post,
                JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {evaluation_run.judgment_api_key}",
                    "X-Organization-Id": evaluation_run.organization_id,
                },
                json=payload,
                verify=True,
            )

            if not response.ok:
                error_message = response.json().get(
                    "detail", "An unknown error occurred."
                )
                error(f"Error adding evaluation to queue: {error_message}")
                raise JudgmentAPIError(error_message)

            info(f"Successfully added evaluation '{evaluation_run.eval_name}' to queue")

            # Poll until the evaluation is complete
            results = await _poll_evaluation_until_complete(
                eval_name=evaluation_run.eval_name,
                project_name=evaluation_run.project_name,
                judgment_api_key=evaluation_run.judgment_api_key,
                organization_id=evaluation_run.organization_id,
                original_examples=evaluation_run.examples,  # Pass the original examples
            )

            pretty_str_to_print = None
            if results:  # Ensure results exist before logging
                send_results = [
                    scoring_result.model_dump(warnings=False)
                    for scoring_result in results
                ]
                try:
                    # Run the blocking log_evaluation_results in a separate thread
                    pretty_str_to_print = await asyncio.to_thread(
                        log_evaluation_results, send_results, evaluation_run
                    )
                except Exception as e:
                    error(f"Error logging results after async evaluation: {str(e)}")

            return results, pretty_str_to_print

        # Create a regular task
        task = asyncio.create_task(_async_evaluation_workflow())

        # Wrap it in our custom awaitable that will show a spinner only when awaited
        return SpinnerWrappedTask(
            task, f"Processing evaluation '{evaluation_run.eval_name}': "
        )
    else:
        check_examples(evaluation_run.examples, evaluation_run.scorers)
        if judgment_scorers:
            # Execute evaluation using Judgment API
            info("Starting API evaluation")
            debug(f"Creating API evaluation run with {len(judgment_scorers)} scorers")
            try:  # execute an EvaluationRun with just JudgmentScorers
                api_evaluation_run: EvaluationRun = EvaluationRun(
                    eval_name=evaluation_run.eval_name,
                    project_name=evaluation_run.project_name,
                    examples=evaluation_run.examples,
                    scorers=judgment_scorers,
                    model=evaluation_run.model,
                    judgment_api_key=evaluation_run.judgment_api_key,
                    organization_id=evaluation_run.organization_id,
                )
                debug("Sending request to Judgment API")
                response_data: Dict = run_with_spinner(
                    "Running Evaluation: ", execute_api_eval, api_evaluation_run
                )
                info(f"Received {len(response_data['results'])} results from API")
            except JudgmentAPIError as e:
                error(
                    f"An error occurred while executing the Judgment API request: {str(e)}"
                )
                raise JudgmentAPIError(
                    f"An error occurred while executing the Judgment API request: {str(e)}"
                )
            except ValueError as e:
                raise ValueError(
                    f"Please check your EvaluationRun object, one or more fields are invalid: {str(e)}"
                )

            # Convert the response data to `ScoringResult` objects
            debug("Processing API results")
            api_results = [
                ScoringResult(**result) for result in response_data["results"]
            ]
        # Run local evals
        if local_scorers:  # List[BaseScorer]
            # We should be removing local scorers soon
            info("Starting local evaluation")
            for example in evaluation_run.examples:
                with example_logging_context(example.created_at, example.example_id):
                    debug(f"Processing example {example.example_id}: {example.input}")

            results: List[ScoringResult] = safe_run_async(
                a_execute_scoring(
                    evaluation_run.examples,
                    local_scorers,
                    model=evaluation_run.model,
                    skip_on_missing_params=True,
                    show_indicator=True,
                    _use_bar_indicator=True,
                    throttle_value=0,
                    max_concurrent=MAX_CONCURRENT_EVALUATIONS,
                )
            )
            local_results = results
            info(f"Local evaluation complete with {len(local_results)} results")
        # Aggregate the ScorerData from the API and local evaluations
        debug("Merging API and local results")
        merged_results: List[ScoringResult] = merge_results(api_results, local_results)
        merged_results = check_missing_scorer_data(merged_results)

        info(f"Successfully merged {len(merged_results)} results")

        # Evaluate rules against local scoring results if rules exist (this cant be done just yet)
        # if evaluation_run.rules and merged_results:
        #     run_rules(
        #         local_results=merged_results,
        #         rules=evaluation_run.rules,
        #         judgment_api_key=evaluation_run.judgment_api_key,
        #         organization_id=evaluation_run.organization_id
        #     )
        # print(merged_results)
        send_results = [
            scoring_result.model_dump(warnings=False)
            for scoring_result in merged_results
        ]
        pretty_str = run_with_spinner(
            "Logging Results: ",
            log_evaluation_results,
            send_results,
            evaluation_run,
        )
        rprint(pretty_str)

        for i, result in enumerate(merged_results):
            if (
                not result.scorers_data
            ):  # none of the scorers could be executed on this example
                info(
                    f"None of the scorers could be executed on example {i}. This is usually because the Example is missing the fields needed by the scorers. Try checking that the Example has the necessary fields for your scorers."
                )
        return merged_results


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
            # error_msg += f"\nInput: {fail_case['input']}\n"
            # error_msg += f"Actual Output: {fail_case['actual_output']}\n"
            # error_msg += f"Expected Output: {fail_case['expected_output']}\n"
            # error_msg += f"Context: {fail_case['context']}\n"
            # error_msg += f"Retrieval Context: {fail_case['retrieval_context']}\n"
            # error_msg += f"Additional Metadata: {fail_case['additional_metadata']}\n"
            # error_msg += f"Tools Called: {fail_case['tools_called']}\n"
            # error_msg += f"Expected Tools: {fail_case['expected_tools']}\n"

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
                    f"Evaluation Cost: {fail_scorer.evaluation_cost}\n"
                    f"Verbose Logs: {fail_scorer.verbose_logs}\n"
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
