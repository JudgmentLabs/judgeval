"""
Infrastructure for executing evaluations of `Example`s using one or more `BaseScorer`s.
"""

import asyncio
import time
from tqdm.asyncio import tqdm_asyncio
from typing import List, Union, Optional, Callable
from rich.progress import Progress, SpinnerColumn, TextColumn

from judgeval.data import (
    Example,
    CustomExample,
    ScoringResult,
    generate_scoring_result,
    create_scorer_data,
)
from judgeval.scorers import BaseScorer
from judgeval.scorers.utils import clone_scorers, scorer_console_msg
from judgeval.common.exceptions import MissingTestCaseParamsError
from judgeval.common.logger import example_logging_context, debug, error, warning, info
from judgeval.judges import JudgevalJudge


async def safe_a_score_example(
    scorer: BaseScorer,
    example: Example,
    ignore_errors: bool,
    skip_on_missing_params: bool,
):
    """
    Scoring task function when not using a progress indicator!
    "Safely" scores an `Example` using a `BaseScorer` by gracefully handling any exceptions that may occur.

    Args:
        scorer (BaseScorer): The `BaseScorer` to use for scoring the example.
        example (Example): The `Example` to be scored.

        ignore_errors (bool): Whether to ignore errors during the evaluation.
        If set to false, any error will be raised and stop the evaluation.
        If set to true, the error will be stored in the `error` attribute of the `BaseScorer` and the `success` attribute will be set to False.

        skip_on_missing_params (bool): Whether to skip the test case if required parameters are missing.
    """
    debug(f"Starting safe_a_score_example for example {example.example_id}")
    try:
        await scorer.a_score_example(example, _show_indicator=False)
        info(f"Successfully scored example {example.example_id}")
    except MissingTestCaseParamsError as e:
        if (
            skip_on_missing_params
        ):  # Skip the example if the scorer requires parameters that are missing
            with example_logging_context(example.created_at, example.example_id):
                warning(
                    f"Skipping example {example.example_id} due to missing parameters"
                )
            scorer.skipped = True
            return
        else:
            if (
                ignore_errors
            ):  # Gracefully handle the error, does not stop the evaluation
                scorer.error = str(e)
                scorer.success = False
                with example_logging_context(example.created_at, example.example_id):
                    warning(
                        f"Ignoring errors for example {example.example_id}: {str(e)} due to missing parameters"
                    )
            else:  # Raise the error and stop the evaluation
                with example_logging_context(example.created_at, example.example_id):
                    error(
                        f"Stopping example {example.example_id}: {str(e)} due to missing parameters"
                    )
                raise
    except TypeError:  # in case a_score_example does not accept _show_indicator
        try:
            await scorer.a_score_example(example)
        except MissingTestCaseParamsError as e:
            if skip_on_missing_params:
                scorer.skipped = True
                with example_logging_context(example.created_at, example.example_id):
                    warning(
                        f"Skipping example {example.example_id} due to missing parameters"
                    )
                return
            else:
                if ignore_errors:
                    scorer.error = str(e)
                    scorer.success = False
                    with example_logging_context(
                        example.created_at, example.example_id
                    ):
                        warning(
                            f"Ignoring errors for example {example.example_id}: {str(e)} due to missing parameters"
                        )
                else:
                    with example_logging_context(
                        example.created_at, example.example_id
                    ):
                        error(
                            f"Stopping example {example.example_id}: {str(e)} due to missing parameters"
                        )
                    raise
    except Exception as e:
        if ignore_errors:
            scorer.error = str(e)
            scorer.success = False  # Assuming you want to set success to False
            with example_logging_context(example.created_at, example.example_id):
                warning(f"Ignoring errors for example {example.example_id}: {str(e)}")
        else:
            with example_logging_context(example.created_at, example.example_id):
                error(f"Stopping example {example.example_id}: {str(e)}")
            raise


async def score_task(
    task_id: int,
    progress: Progress,
    scorer: BaseScorer,
    example: Example,
    ignore_errors: bool = True,
    skip_on_missing_params: bool = True,
):
    """
    Task function for asynchronously measuring a given example using a BaseScorer.

    Args:
        task_id (int): The ID of the task being measured.
        progress (Progress): An instance of the Progress class to track task progress.
        scorer (BaseScorer): An instance of the BaseScorer class used to score the example.
        example (Example): The example to be scored.
        ignore_errors (bool, optional): Whether to ignore errors during scoring. Defaults to True.
        skip_on_missing_params (bool, optional): Whether to skip scoring if there are missing parameters. Defaults to True.

    Raises:
        MissingTestCaseParamsError: If required test case parameters are missing and skip_on_missing_params is False.
        Exception: If an unexpected error occurs and ignore_errors is False.

    Returns:
        None
    """
    while not progress.finished:
        start_time = time.perf_counter()

        try:
            await scorer.a_score_example(example, _show_indicator=False)
            finish_text = "Completed"
        except MissingTestCaseParamsError as e:
            if skip_on_missing_params:
                scorer.skipped = True
                with example_logging_context(example.created_at, example.example_id):
                    debug(
                        f"Skipping example {example.example_id} due to missing parameters"
                    )
                return
            else:
                if ignore_errors:
                    scorer.error = str(e)
                    scorer.success = False  # Override success
                    finish_text = "Failed"
                else:
                    with example_logging_context(
                        example.created_at, example.example_id
                    ):
                        error(
                            f"Stopping example {example.example_id}: {str(e)} due to missing parameters"
                        )
                    raise
        except TypeError:
            try:
                await scorer.a_score_example(example)
                finish_text = "Completed"
            except MissingTestCaseParamsError as e:
                if skip_on_missing_params:
                    scorer.skipped = True
                    with example_logging_context(
                        example.created_at, example.example_id
                    ):
                        debug(
                            f"Skipping example {example.example_id} due to missing parameters"
                        )
                    return
                else:
                    if ignore_errors:
                        scorer.error = str(e)
                        scorer.success = False  # Override success
                        finish_text = "Failed"
                    else:
                        with example_logging_context(
                            example.created_at, example.example_id
                        ):
                            error(
                                f"Stopping example {example.example_id}: {str(e)} due to missing parameters"
                            )
                        raise
        except Exception as e:
            if ignore_errors:
                scorer.error = str(e)
                scorer.success = False  # Override success
                finish_text = "Failed"
                with example_logging_context(example.created_at, example.example_id):
                    warning(
                        f"Ignoring errors for example {example.example_id}: {str(e)}"
                    )
            else:
                with example_logging_context(example.created_at, example.example_id):
                    error(f"Stopping example {example.example_id}: {str(e)}")
                raise

        end_time = time.perf_counter()
        time_taken = format(end_time - start_time, ".2f")
        progress.update(task_id, advance=100)  # Mark task as complete
        progress.update(
            task_id,
            description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]{finish_text}! ({time_taken}s)",
        )
        break


async def score_with_indicator(
    scorers: List[BaseScorer],
    example: Example,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
):
    """
    Scores an example using a list of BaseScorers, optionally displaying a progress indicator.

    Args:
        scorers (List[BaseScorer]): A list of BaseScorer objects to evaluate the example.
        example (Example): The example to be scored.
        ignore_errors (bool): If True, errors during scoring will be ignored.
        skip_on_missing_params (bool): If True, scoring will be skipped if required parameters are missing.
        show_indicator (bool): If True, a progress indicator will be displayed during scoring.

    Returns:
        None

    Raises:
        Any exceptions raised by the scoring functions, unless `ignore_errors` is True.
    """
    if show_indicator:
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            tasks = []
            for scorer in scorers:
                task_id = progress.add_task(
                    description=scorer_console_msg(scorer, async_mode=True),
                    total=100,
                )  # Add task to progress bar
                tasks.append(
                    score_task(
                        task_id,
                        progress,
                        scorer,
                        example,
                        ignore_errors,
                        skip_on_missing_params,
                    )  # Create and execute task to score the example with a single scorer
                )
            await asyncio.gather(*tasks)
    else:
        tasks = [
            safe_a_score_example(scorer, example, ignore_errors, skip_on_missing_params)
            for scorer in scorers
        ]

        await asyncio.gather(*tasks)


async def a_execute_scoring(
    examples: Union[List[Example], List[CustomExample]],
    scorers: List[BaseScorer],
    model: Optional[Union[str, List[str], JudgevalJudge]] = "gpt-4.1",
    ignore_errors: bool = True,
    skip_on_missing_params: bool = True,
    show_indicator: bool = True,
    throttle_value: int = 0,
    max_concurrent: int = 100,
    verbose_mode: Optional[bool] = None,
    _use_bar_indicator: bool = True,
) -> List[ScoringResult]:
    """
    Executes evaluations of `Example`s asynchronously using one or more `BaseScorer`s.
    Each `Example` will be evaluated by all of the `BaseScorer`s in the `scorers` list.

    Args:
        examples (Union[List[Example], List[CustomExample]]): A list of `Example` objects to be evaluated.
        scorers (List[BaseScorer]): A list of `BaseScorer` objects to evaluate the examples.
        model (Union[str, List[str], JudgevalJudge]): The model to use for evaluation.
        ignore_errors (bool): Whether to ignore errors during evaluation.
        skip_on_missing_params (bool): Whether to skip evaluation if parameters are missing.
        show_indicator (bool): Whether to show a progress indicator.
        throttle_value (int): The amount of time to wait between starting each task.
        max_concurrent (int): The maximum number of concurrent tasks.
        verbose_mode (Optional[bool]): If set, enables verbose mode for scorers.
        _use_bar_indicator (bool): Whether to use a progress bar indicator.

    Returns:
        List[ScoringResult]: A list of `ScoringResult` objects containing the evaluation results.
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"Error executing function: {e}")
                if kwargs.get("ignore_errors", False):
                    # Simply return None when ignoring errors, as expected by the test
                    return None
                # If we're not ignoring errors, propagate the exception
                raise

    if verbose_mode is not None:
        for scorer in scorers:
            scorer.verbose_mode = verbose_mode

    # Add model to scorers
    for scorer in scorers:
        scorer._add_model(model)

    scoring_results: List[ScoringResult] = [None for _ in examples]
    tasks = []
    cloned_scorers: List[BaseScorer]

    if show_indicator and _use_bar_indicator:
        with tqdm_asyncio(
            desc=f"Evaluating {len(examples)} example(s) in parallel",
            unit="Example",
            total=len(examples),
            bar_format="{desc}: |{bar}|{percentage:3.0f}% ({n_fmt}/{total_fmt}) [Time Taken: {elapsed}, {rate_fmt}{postfix}]",
        ) as pbar:
            for i, ex in enumerate(examples):
                with example_logging_context(ex.created_at, ex.example_id):
                    debug(f"Starting scoring for example {ex.example_id}")
                    debug(f"Input: {ex.input}")
                    debug(f"Using {len(scorers)} scorers")
                    for scorer in scorers:
                        debug(f"Using scorer: {type(scorer).__name__}")
                        if hasattr(scorer, "threshold"):
                            debug(f"Scorer threshold: {scorer.threshold}")
                        if hasattr(scorer, "model"):
                            debug(f"Scorer model: {type(scorer.model).__name__}")
                if isinstance(ex, Example) or isinstance(ex, CustomExample):
                    if len(scorers) == 0:
                        pbar.update(1)
                        continue

                    cloned_scorers = clone_scorers(scorers)
                    task = execute_with_semaphore(
                        func=a_eval_examples_helper,
                        scorers=cloned_scorers,
                        example=ex,
                        scoring_results=scoring_results,
                        score_index=i,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        show_indicator=show_indicator,
                        _use_bar_indicator=_use_bar_indicator,
                        pbar=pbar,
                    )
                    tasks.append(asyncio.create_task(task))

                await asyncio.sleep(throttle_value)
            await asyncio.gather(*tasks)
    else:
        for i, ex in enumerate(examples):
            if isinstance(ex, Example) or isinstance(ex, CustomExample):
                if len(scorers) == 0:
                    continue

                cloned_scorers = clone_scorers(scorers)
                task = execute_with_semaphore(
                    func=a_eval_examples_helper,
                    scorers=cloned_scorers,
                    example=ex,
                    scoring_results=scoring_results,
                    score_index=i,
                    ignore_errors=ignore_errors,
                    skip_on_missing_params=skip_on_missing_params,
                    _use_bar_indicator=_use_bar_indicator,
                    show_indicator=show_indicator,
                )
                tasks.append(asyncio.create_task((task)))

            await asyncio.sleep(throttle_value)
        await asyncio.gather(*tasks)
    return scoring_results


async def a_eval_examples_helper(
    scorers: List[BaseScorer],
    example: Union[Example, CustomExample],
    scoring_results: List[ScoringResult],
    score_index: int,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    pbar: Optional[tqdm_asyncio] = None,
) -> None:
    """
    Evaluate a single example asynchronously using a list of scorers.

    Args:
        scorers (List[BaseScorer]): List of BaseScorer objects to evaluate the example.
        example (Example): The example to be evaluated.
        scoring_results (List[ScoringResult]): List to store the scoring results.
        score_index (int): Index at which the result should be stored in scoring_results.
        ignore_errors (bool): Flag to indicate whether to ignore errors during scoring.
        skip_on_missing_params (bool): Flag to indicate whether to skip scoring if parameters are missing.
        show_indicator (bool): Flag to indicate whether to show a progress indicator.
        _use_bar_indicator (bool): Flag to indicate whether to use a bar indicator for progress.
        pbar (Optional[tqdm_asyncio]): Optional progress bar for tracking progress.
    Returns:
        None
    """

    show_metrics_indicator = show_indicator and not _use_bar_indicator

    for scorer in scorers:
        scorer.skipped = False
        scorer.error = None  # Reset scorer error

    # scoring the Example
    scoring_start_time = time.perf_counter()
    await score_with_indicator(
        scorers=scorers,
        example=example,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
    )  # execute the scoring functions of each scorer on the example

    # Now that all the scoring functions of each scorer have executed, we collect
    # the results and update the ScoringResult with the scorer data
    success = True
    scorer_data_list = []
    for scorer in scorers:
        # At this point, the scorer has been executed and already contains data.
        if getattr(scorer, "skipped", False):
            continue
        scorer_data = create_scorer_data(
            scorer
        )  # Fetch scorer data from completed scorer evaluation
        for s in scorer_data:
            success = success and s.success
        scorer_data_list.extend(scorer_data)

    scoring_end_time = time.perf_counter()
    run_duration = scoring_end_time - scoring_start_time

    scoring_result = generate_scoring_result(
        example, scorer_data_list, run_duration, success
    )
    scoring_results[score_index] = scoring_result

    if pbar is not None:
        pbar.update(1)
