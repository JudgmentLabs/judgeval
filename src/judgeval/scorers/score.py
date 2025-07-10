"""
Infrastructure for executing evaluations of `Example`s using one or more `BaseScorer`s.
"""

import asyncio
import time
from tqdm.asyncio import tqdm_asyncio
from typing import List, Union, Optional, Callable

from judgeval.data import (
    Example,
    ScoringResult,
    generate_scoring_result,
    create_scorer_data,
)
from judgeval.scorers import BaseScorer
from judgeval.scorers.utils import clone_scorers
from judgeval.common.logger import judgeval_logger
from judgeval.judges import JudgevalJudge


async def safe_a_score_example(
    scorer: BaseScorer,
    example: Example,
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
    try:
        scorer.score = await scorer.a_score_example(example)
        scorer.success = scorer.success_check()
    except Exception as e:
        judgeval_logger.error(f"Error during scoring: {str(e)}")
        scorer.error = str(e)
        scorer.success = False
        return


async def a_execute_scoring(
    examples: List[Example],
    scorers: List[BaseScorer],
    model: Optional[Union[str, List[str], JudgevalJudge]] = "gpt-4.1",
    ignore_errors: bool = False,
    throttle_value: int = 0,
    max_concurrent: int = 100,
) -> List[ScoringResult]:
    """
    Executes evaluations of `Example`s asynchronously using one or more `BaseScorer`s.
    Each `Example` will be evaluated by all of the `BaseScorer`s in the `scorers` list.

    Args:
        examples (List[Example]): A list of `Example` objects to be evaluated.
        scorers (List[BaseScorer]): A list of `BaseScorer` objects to evaluate the examples.
        model (Union[str, List[str], JudgevalJudge]): The model to use for evaluation.
        ignore_errors (bool): Whether to ignore errors during evaluation.
        throttle_value (int): The amount of time to wait between starting each task.
        max_concurrent (int): The maximum number of concurrent tasks.

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
                judgeval_logger.error(f"Error executing function: {e}")
                if kwargs.get("ignore_errors", False):
                    # Simply return None when ignoring errors, as expected by the test
                    return None
                # If we're not ignoring errors, propagate the exception
                raise

    # Add model to scorers
    for scorer in scorers:
        scorer._add_model(model)

    scoring_results: List[ScoringResult] = [None for _ in examples]
    tasks = []
    cloned_scorers: List[BaseScorer]

    with tqdm_asyncio(
        desc=f"Evaluating {len(examples)} example(s) in parallel",
        unit="Example",
        total=len(examples),
        bar_format="{desc}: |{bar}|{percentage:3.0f}% ({n_fmt}/{total_fmt}) [Time Taken: {elapsed}, {rate_fmt}{postfix}]",
    ) as pbar:
        for i, ex in enumerate(examples):
            if isinstance(ex, Example):
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
                    pbar=pbar,
                )
                tasks.append(asyncio.create_task(task))

            await asyncio.sleep(throttle_value)
        await asyncio.gather(*tasks)
    return scoring_results


async def a_eval_examples_helper(
    scorers: List[BaseScorer],
    example: Example,
    scoring_results: List[ScoringResult],
    score_index: int,
    ignore_errors: bool,
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
        pbar (Optional[tqdm_asyncio]): Optional progress bar for tracking progress.
    Returns:
        None
    """

    # scoring the Example
    scoring_start_time = time.perf_counter()

    tasks = [safe_a_score_example(scorer, example) for scorer in scorers]

    await asyncio.gather(*tasks)

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
