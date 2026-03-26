from __future__ import annotations

from typing import List, Optional, Union

from judgeval.logger import judgeval_logger
from judgeval.data.example import Example
from judgeval.data.scoring_result import ScoringResult
from judgeval.judges import Judge
from judgeval.internal.api import JudgmentSyncClient
from judgeval.evaluation.local_evaluation import LocalEvaluatorRunner
from judgeval.evaluation.hosted_evaluation import HostedEvaluatorRunner


class Evaluation:
    """Score a batch of examples using hosted scorers or custom judges.

    Create an `Evaluation` via `client.evaluation.create()`, then call
    `.run()` to execute scorers against your examples.

    Two modes are supported:

    - **Hosted scorers** -- pass scorer names as strings (e.g.
      `"faithfulness"`, `"answer_relevancy"`). Evaluation runs server-side
      on the Judgment platform.
    - **Custom judges** -- pass `Judge` subclass instances for in-process
      evaluation with your own scoring logic.

    Examples:
        Using hosted scorers:

        ```python
        evaluation = client.evaluation.create()
        results = evaluation.run(
            examples=examples,
            scorers=["faithfulness", "answer_relevancy"],
            eval_run_name="nightly-eval",
        )
        for result in results:
            print(result.success, result.scorers_data)
        ```

        Using a custom judge:

        ```python
        evaluation = client.evaluation.create()
        results = evaluation.run(
            examples=examples,
            scorers=[ToxicityJudge()],
            eval_run_name="toxicity-check",
        )
        ```
    """

    __slots__ = ("_local", "_hosted")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._local = LocalEvaluatorRunner(client, project_id, project_name)
        self._hosted = HostedEvaluatorRunner(client, project_id, project_name)

    def run(
        self,
        examples: List[Example],
        scorers: Union[List[str], List[Judge]],
        eval_run_name: str,
        assert_test: bool = False,
        timeout_seconds: int = 300,
    ) -> List[ScoringResult]:
        """Run scorers against your examples and return results.

        Pass **either** hosted scorer names (strings) **or** custom `Judge`
        instances. Mixing both in one call is not supported.

        Args:
            examples: The `Example` objects to evaluate.
            scorers: Hosted scorer names (e.g. `["faithfulness"]`) or
                `Judge` instances (e.g. `[ToxicityJudge()]`).
            eval_run_name: A name for this run, visible in the dashboard.
            assert_test: If True, raises an exception when any scorer
                fails its threshold. Useful in CI/CD pipelines.
            timeout_seconds: Maximum seconds to wait for hosted scorer
                results before timing out.

        Returns:
            A list of `ScoringResult` objects, one per example.

        Examples:
            ```python
            results = evaluation.run(
                examples=[
                    Example.create(
                        input="What is Python?",
                        actual_output="A programming language.",
                        expected_output="A high-level programming language.",
                    ),
                ],
                scorers=["answer_relevancy"],
                eval_run_name="quick-test",
            )
            print(results[0].success)  # True/False
            ```
        """
        hosted_scorers = [s for s in scorers if isinstance(s, str)]
        local_scorers = [s for s in scorers if isinstance(s, Judge)]
        if len(local_scorers) > 0 and len(hosted_scorers) > 0:
            judgeval_logger.error(
                "Running both local and hosted scorers is not supported. "
                "Please run your evaluation with either local or hosted scorers, but not both."
            )
            return []
        if len(local_scorers) == 0 and len(hosted_scorers) == 0:
            judgeval_logger.error(
                "No valid local or hosted scorers provided. "
                "Please provide at least one local or hosted scorer."
            )
            return []
        if len(local_scorers) > 0:
            return self._local.run(
                examples,
                local_scorers,
                eval_run_name,
                assert_test=assert_test,
                timeout_seconds=timeout_seconds,
            )
        if len(hosted_scorers) > 0:
            return self._hosted.run(
                examples,
                hosted_scorers,
                eval_run_name,
                assert_test=assert_test,
                timeout_seconds=timeout_seconds,
            )
        return []
