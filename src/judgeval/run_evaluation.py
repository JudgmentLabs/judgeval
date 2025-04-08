import asyncio
import requests
import time
import sys
import itertools
import threading
from typing import List, Dict, Any
from datetime import datetime
from rich import print as rprint

from judgeval.data import (
    ScorerData, 
    ScoringResult,
    Example
)
from judgeval.scorers import (
    JudgevalScorer, 
    APIJudgmentScorer,
    ClassifierScorer
)
from judgeval.scorers.score import a_execute_scoring
from judgeval.constants import (
    ROOT_API,
    JUDGMENT_EVAL_API_URL,
    JUDGMENT_EVAL_LOG_API_URL,
    MAX_CONCURRENT_EVALUATIONS,
    JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL
)
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.common.logger import (
    debug, 
    info, 
    error, 
    example_logging_context
)
from judgeval.evaluation_run import EvaluationRun


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
            "X-Organization-Id": evaluation_run.organization_id
        },   
        json=payload,
        verify=True
    )
    return response.json()

def execute_api_eval(evaluation_run: EvaluationRun) -> List[Dict]:
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
        payload = evaluation_run.model_dump(warnings=False)
        response = requests.post(
            JUDGMENT_EVAL_API_URL, 
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {evaluation_run.judgment_api_key}",
                "X-Organization-Id": evaluation_run.organization_id
            }, 
            json=payload,
            verify=True
        )
        response_data = response.json()
    except Exception as e:
        error(f"Error: {e}")
        details = response.json().get("detail", "No details provided")
        raise JudgmentAPIError("An error occurred while executing the Judgment API request: " + details)
    # Check if the response status code is not 2XX
    # Add check for the duplicate eval run name
    if not response.ok:
        error_message = response_data.get('detail', 'An unknown error occurred.')
        error(f"Error: {error_message=}")
        raise JudgmentAPIError(error_message)
    return response_data


def merge_results(api_results: List[ScoringResult], local_results: List[ScoringResult]) -> List[ScoringResult]:
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
        return api_results
    if not api_results and local_results:
        return local_results

    if len(api_results) != len(local_results):
        # Results should be of same length because each ScoringResult is a 1-1 mapping to an Example
        raise ValueError(f"The number of API and local results do not match: {len(api_results)} vs {len(local_results)}")
    
    # Each ScoringResult in api and local have all the same fields besides `scorers_data`
    for api_result, local_result in zip(api_results, local_results):
        if not (api_result.data_object and local_result.data_object):
            raise ValueError("Data object is None in one of the results.")
        if api_result.data_object.input != local_result.data_object.input:
            raise ValueError("The API and local results are not aligned.")
        if api_result.data_object.actual_output != local_result.data_object.actual_output:
            raise ValueError("The API and local results are not aligned.")
        if api_result.data_object.expected_output != local_result.data_object.expected_output:
            raise ValueError("The API and local results are not aligned.")
        if api_result.data_object.context != local_result.data_object.context:
            raise ValueError("The API and local results are not aligned.")
        if api_result.data_object.retrieval_context != local_result.data_object.retrieval_context:
            raise ValueError("The API and local results are not aligned.")
        if api_result.data_object.additional_metadata != local_result.data_object.additional_metadata:
            raise ValueError("The API and local results are not aligned.")
        if api_result.data_object.tools_called != local_result.data_object.tools_called:
            raise ValueError("The API and local results are not aligned.")
        if api_result.data_object.expected_tools != local_result.data_object.expected_tools:
            raise ValueError("The API and local results are not aligned.")
        
        
        # Merge ScorerData from the API and local scorers together
        api_scorer_data = api_result.scorers_data
        local_scorer_data = local_result.scorers_data
        if api_scorer_data is None and local_scorer_data is not None:
            api_result.scorers_data = local_scorer_data

        if api_scorer_data is not None and local_scorer_data is not None:
            api_result.scorers_data = api_scorer_data + local_scorer_data
    
    return api_results


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


def check_eval_run_name_exists(eval_name: str, project_name: str, judgment_api_key: str, organization_id: str) -> None:
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
                "X-Organization-Id": organization_id
            },
            json={
                "eval_name": eval_name,
                "project_name": project_name,
                "judgment_api_key": judgment_api_key,
            },
            verify=True
        )
        
        if response.status_code == 409:
            error(f"Eval run name '{eval_name}' already exists for this project. Please choose a different name or set the `override` flag to true.")
            raise ValueError(f"Eval run name '{eval_name}' already exists for this project. Please choose a different name or set the `override` flag to true.")
        
        if not response.ok:
            response_data = response.json()
            error_message = response_data.get('detail', 'An unknown error occurred.')
            error(f"Error checking eval run name: {error_message}")
            raise JudgmentAPIError(error_message)
            
    except requests.exceptions.RequestException as e:
        error(f"Failed to check if eval run name exists: {str(e)}")
        raise JudgmentAPIError(f"Failed to check if eval run name exists: {str(e)}")


def log_evaluation_results(merged_results: List[ScoringResult], evaluation_run: EvaluationRun) -> str:
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
                "Authorization": f"Bearer {evaluation_run.judgment_api_key}",
                "X-Organization-Id": evaluation_run.organization_id
            },
            json={
                "results": [result.to_dict() for result in merged_results],
                "project_name": evaluation_run.project_name,
                "eval_name": evaluation_run.eval_name,
            },
            verify=True
        )
        
        if not res.ok:
            response_data = res.json()
            error_message = response_data.get('detail', 'An unknown error occurred.')
            error(f"Error {res.status_code}: {error_message}")
            raise JudgmentAPIError(error_message)
        
        if "ui_results_url" in res.json():
            url = res.json()['ui_results_url']
            pretty_str = f"\n🔍 You can view your evaluation results here: [rgb(106,0,255)][link={url}]View Results[/link]\n"
            return pretty_str
            
    except requests.exceptions.RequestException as e:
        error(f"Request failed while saving evaluation results to DB: {str(e)}")
        raise JudgmentAPIError(f"Request failed while saving evaluation results to DB: {str(e)}")
    except Exception as e:
        error(f"Failed to save evaluation results to DB: {str(e)}")
        raise ValueError(f"Failed to save evaluation results to DB: {str(e)}")

def run_with_spinner(message: str, func, *args, **kwargs) -> Any:
        """Run a function with a spinner in the terminal."""
        spinner = itertools.cycle(['|', '/', '-', '\\'])

        def display_spinner():
            while not stop_spinner_event.is_set():
                sys.stdout.write(f'\r{message}{next(spinner)}')
                sys.stdout.flush() 
                time.sleep(0.1)

        stop_spinner_event = threading.Event()
        spinner_thread = threading.Thread(target=display_spinner)
        spinner_thread.start()

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error(f"An error occurred: {str(e)}")
            stop_spinner_event.set()
            spinner_thread.join()
            raise e
        finally:
            stop_spinner_event.set()
            spinner_thread.join()

            sys.stdout.write('\r' + ' ' * (len(message) + 1) + '\r')
            sys.stdout.flush()

        return result

def check_examples(examples: List[Example], scorers: List[APIJudgmentScorer]) -> None:
    """
    Checks if the example contains the necessary parameters for the scorer.
    """
    for scorer in scorers:
        if isinstance(scorer, APIJudgmentScorer):
            for example in examples:
                missing_params = []
                for param in scorer.required_params:
                    if getattr(example, param.value) is None:
                        missing_params.append(f"'{param.value}'")
                if missing_params:
                    # We do this because we want to inform users that an example is missing parameters for a scorer
                    # Example ID (usually random UUID) does not provide any helpful information for the user but printing the entire example is overdoing it
                    print(f"WARNING: Example {example.example_id} is missing the following parameters: {missing_params} for scorer {scorer.score_type.value}")

def run_eval(evaluation_run: EvaluationRun, override: bool = False, ignore_errors: bool = True, async_execution: bool = False) -> List[ScoringResult]:
    """
    Executes an evaluation of `Example`s using one or more `Scorer`s

    Args:
        evaluation_run (EvaluationRun): Stores example and evaluation together for running
        override (bool, optional): Whether to override existing evaluation run with same name. Defaults to False.
        ignore_errors (bool, optional): Whether to ignore scorer errors during evaluation. Defaults to True.
    
        Args: 
            project_name (str): The name of the project the evaluation results belong to
            eval_name (str): The name of the evaluation run
            examples (List[Example]): The examples to evaluate
            scorers (List[Union[JudgmentScorer, JudgevalScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
            metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
            judgment_api_key (Optional[str]): The API key for running evaluations on the Judgment API
            log_results (bool): Whether to log the results to the Judgment API
            rules (Optional[List[Rule]]): Rules to evaluate against scoring results

    Returns:
        List[ScoringResult]: The results of the evaluation. Each result is a dictionary containing the fields of a `ScoringResult` object.
    """

    # Call endpoint to check to see if eval run name exists (if we DON'T want to override and DO want to log results)
    if not override and evaluation_run.log_results:
        check_eval_run_name_exists(
            evaluation_run.eval_name,
            evaluation_run.project_name,
            evaluation_run.judgment_api_key,
            evaluation_run.organization_id
        )
    
    # Set example IDs if not already set
    debug("Initializing examples with IDs and timestamps")
    for idx, example in enumerate(evaluation_run.examples):
        example.example_index = idx  # Set numeric index
        example.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with example_logging_context(example.timestamp, example.example_id):
            debug(f"Initialized example {example.example_id} (index: {example.example_index})")
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
    
    # Group APIJudgmentScorers and JudgevalScorers, then evaluate them in parallel
    debug("Grouping scorers by type")
    judgment_scorers: List[APIJudgmentScorer] = []
    local_scorers: List[JudgevalScorer] = []
    for scorer in evaluation_run.scorers:
        if isinstance(scorer, (APIJudgmentScorer, ClassifierScorer)):
            judgment_scorers.append(scorer)
            debug(f"Added judgment scorer: {type(scorer).__name__}")
        else:
            local_scorers.append(scorer)
            debug(f"Added local scorer: {type(scorer).__name__}")
    
    debug(f"Found {len(judgment_scorers)} judgment scorers and {len(local_scorers)} local scorers")
    
    api_results: List[ScoringResult] = []
    local_results: List[ScoringResult] = []

    if async_execution:
        check_examples(evaluation_run.examples, evaluation_run.scorers)
        info("Starting async evaluation")
        payload = evaluation_run.model_dump(warnings=False)
        requests.post(
            JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {evaluation_run.judgment_api_key}",
                "X-Organization-Id": evaluation_run.organization_id
            },
            json=payload,
            verify=True
        )
        print("Successfully added evaluation to queue")
    else:
        if judgment_scorers:
            # Execute evaluation using Judgment API
            check_examples(evaluation_run.examples, evaluation_run.scorers)
            info("Starting API evaluation")
            debug(f"Creating API evaluation run with {len(judgment_scorers)} scorers")
            try:  # execute an EvaluationRun with just JudgmentScorers
                api_evaluation_run: EvaluationRun = EvaluationRun(
                    eval_name=evaluation_run.eval_name,
                    project_name=evaluation_run.project_name,
                    examples=evaluation_run.examples,
                    scorers=judgment_scorers,
                    model=evaluation_run.model,
                    aggregator=evaluation_run.aggregator,
                    metadata=evaluation_run.metadata,
                    judgment_api_key=evaluation_run.judgment_api_key,
                    organization_id=evaluation_run.organization_id,
                    log_results=evaluation_run.log_results,
                    rules=evaluation_run.rules
                )
                debug("Sending request to Judgment API")    
                response_data: List[Dict] = run_with_spinner("Running Evaluation: ", execute_api_eval, api_evaluation_run)
                info(f"Received {len(response_data['results'])} results from API")
            except JudgmentAPIError as e:
                error(f"An error occurred while executing the Judgment API request: {str(e)}")
                raise JudgmentAPIError(f"An error occurred while executing the Judgment API request: {str(e)}")
            except ValueError as e:
                raise ValueError(f"Please check your EvaluationRun object, one or more fields are invalid: {str(e)}")
            
            # Convert the response data to `ScoringResult` objects
            debug("Processing API results")
            api_results = [ScoringResult(**result) for result in response_data["results"]]
        # Run local evals
        if local_scorers:  # List[JudgevalScorer]
            # We should be removing local scorers soon
            info("Starting local evaluation")
            for example in evaluation_run.examples:
                with example_logging_context(example.timestamp, example.example_id):
                    debug(f"Processing example {example.example_id}: {example.input}")
            
            results: List[ScoringResult] = asyncio.run(
                a_execute_scoring(
                    evaluation_run.examples,
                    local_scorers,
                    model=evaluation_run.model,
                    ignore_errors=ignore_errors,
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
        if evaluation_run.log_results:
            pretty_str = run_with_spinner("Logging Results: ", log_evaluation_results, merged_results, evaluation_run)
            rprint(pretty_str)

        for i, result in enumerate(merged_results):
            if not result.scorers_data:  # none of the scorers could be executed on this example
                info(f"None of the scorers could be executed on example {i}. This is usually because the Example is missing the fields needed by the scorers. Try checking that the Example has the necessary fields for your scorers.")
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
            test_case = {
                'input': result.data_object.input,
                'actual_output': result.data_object.actual_output,
                'expected_output': result.data_object.expected_output,
                'context': result.data_object.context,
                'retrieval_context': result.data_object.retrieval_context,
                'additional_metadata': result.data_object.additional_metadata,
                'tools_called': result.data_object.tools_called,
                'expected_tools': result.data_object.expected_tools,
                'failed_scorers': []
            }
            if result.scorers_data:
                # If the result was not successful, check each scorer_data
                for scorer_data in result.scorers_data:
                    if not scorer_data.success:
                        test_case['failed_scorers'].append(scorer_data)
            failed_cases.append(test_case)

    if failed_cases:
        error_msg = f"The following test cases failed: \n"
        for fail_case in failed_cases:
            error_msg += f"\nInput: {fail_case['input']}\n"
            error_msg += f"Actual Output: {fail_case['actual_output']}\n"
            error_msg += f"Expected Output: {fail_case['expected_output']}\n"
            error_msg += f"Context: {fail_case['context']}\n"
            error_msg += f"Retrieval Context: {fail_case['retrieval_context']}\n"
            error_msg += f"Additional Metadata: {fail_case['additional_metadata']}\n"
            error_msg += f"Tools Called: {fail_case['tools_called']}\n"
            error_msg += f"Expected Tools: {fail_case['expected_tools']}\n"
    
            for fail_scorer in fail_case['failed_scorers']:

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
            error_msg += "-"*100
    
        raise AssertionError(error_msg)
    