"""
Infrastructure for executing evaluations of `Example`s using one or more `ExampleScorer`s.
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
from judgeval.scorers.example_scorer import ExampleScorer
from judgeval.scorers.utils import clone_scorers
from judgeval.logger import judgeval_logger
from judgeval.judges import JudgevalJudge
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL


async def safe_a_score_example(
    scorer: ExampleScorer,
    example: Example,
):
    """
    Scoring task function when not using a progress indicator!
    "Safely" scores an `Example` using a `ExampleScorer` by gracefully handling any exceptions that may occur.

    Args:
        scorer (ExampleScorer): The `ExampleScorer` to use for scoring the example.
        example (Example): The `Example` to be scored.
    """
    try:
        score = await scorer.a_score_example(example)
        if score is None:
            raise Exception("a_score_example need to return a score")
        elif score < 0:
            judgeval_logger.warning("score cannot be less than 0 , setting to 0")
            score = 0
        elif score > 1:
            judgeval_logger.warning("score cannot be greater than 1 , setting to 1")
            score = 1
        else:
            scorer.score = score
        scorer.success = scorer.success_check()
    except Exception as e:
        judgeval_logger.error(f"Error during scoring: {str(e)}")
        scorer.error = str(e)
        scorer.success = False
        scorer.score = 0
        return


async def a_execute_scoring(
    examples: List[Example],
    scorers: List[ExampleScorer],
    model: Optional[Union[str, List[str], JudgevalJudge]] = JUDGMENT_DEFAULT_GPT_MODEL,
    ignore_errors: bool = False,
    throttle_value: int = 0,
    max_concurrent: int = 100,
    show_progress: bool = True,
) -> List[ScoringResult]:
    """
    Executes evaluations of `Example`s asynchronously using one or more `ExampleScorer`s.
    Each `Example` will be evaluated by all of the `ExampleScorer`s in the `scorers` list.

    Args:
        examples (List[Example]): A list of `Example` objects to be evaluated.
        scorers (List[ExampleScorer]): A list of `ExampleScorer` objects to evaluate the examples.
        model (Union[str, List[str], JudgevalJudge]): The model to use for evaluation.
        ignore_errors (bool): Whether to ignore errors during evaluation.
        throttle_value (int): The amount of time to wait between starting each task.
        max_concurrent (int): The maximum number of concurrent tasks.
        show_progress (bool): Whether to show the progress bar indicator.

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
                    return None
                raise

    for scorer in scorers:
        if not scorer.model and isinstance(model, str):
            scorer._add_model(model)

    scoring_results: List[Optional[ScoringResult]] = [None for _ in examples]
    tasks = []

    if show_progress:
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

                    cloned_scorers = clone_scorers(scorers)  # type: ignore
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
    else:
        for i, ex in enumerate(examples):
            if isinstance(ex, Example):
                if len(scorers) == 0:
                    continue

                cloned_scorers = clone_scorers(scorers)  # type: ignore
                task = execute_with_semaphore(
                    func=a_eval_examples_helper,
                    scorers=cloned_scorers,
                    example=ex,
                    scoring_results=scoring_results,
                    score_index=i,
                    ignore_errors=ignore_errors,
                    pbar=None,
                )
                tasks.append(asyncio.create_task(task))

            await asyncio.sleep(throttle_value)
        await asyncio.gather(*tasks)
    return [result for result in scoring_results if result is not None]


async def a_eval_examples_helper(
    scorers: List[ExampleScorer],
    example: Example,
    scoring_results: List[Optional[ScoringResult]],
    score_index: int,
    ignore_errors: bool,
    pbar: Optional[tqdm_asyncio] = None,
) -> None:
    """
    Evaluate a single example asynchronously using a list of scorers.

    Args:
        scorers (List[ExampleScorer]): List of ExampleScorer objects to evaluate the example.
        example (Example): The example to be evaluated.
        scoring_results (List[ScoringResult]): List to store the scoring results.
        score_index (int): Index at which the result should be stored in scoring_results.
        ignore_errors (bool): Flag to indicate whether to ignore errors during scoring.
        pbar (Optional[tqdm_asyncio]): Optional progress bar for tracking progress.
    Returns:
        None
    """

    scoring_start_time = time.perf_counter()

    tasks = [safe_a_score_example(scorer, example) for scorer in scorers]

    await asyncio.gather(*tasks)

    success = True
    scorer_data_list = []
    for scorer in scorers:
        if getattr(scorer, "skipped", False):
            continue
        scorer_data = create_scorer_data(scorer)
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
