from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from judgeval.data.example import Example
from judgeval.hosted.responses import (
    BaseResponse,
)

R = TypeVar("R", bound=BaseResponse)


class Judge(ABC, Generic[R]):
    """Base class for building custom evaluation scorers.

    Subclass `Judge` and implement the `score` method to create your own
    scorer that runs locally. The type parameter `R` determines the response
    format:

    - `Judge[BinaryResponse]` -- pass/fail scoring
    - `Judge[NumericResponse]` -- numeric scoring (e.g. 0.0 to 1.0)
    - `Judge[CategoricalResponse]` -- classification into categories

    Custom judges are passed to `Evaluation.run()` just like hosted scorers.

    Examples:
        A simple binary scorer:

        ```python
        from judgeval.judges import Judge
        from judgeval.hosted.responses import BinaryResponse

        class ToxicityJudge(Judge[BinaryResponse]):
            async def score(self, data: Example) -> BinaryResponse:
                output = data["actual_output"]
                is_clean = not any(word in output for word in BLOCKED_WORDS)
                return BinaryResponse(
                    value=is_clean,
                    reason="No blocked words found" if is_clean else "Contains blocked content",
                )
        ```

        A numeric scorer:

        ```python
        from judgeval.hosted.responses import NumericResponse

        class LengthScorer(Judge[NumericResponse]):
            async def score(self, data: Example) -> NumericResponse:
                output = data["actual_output"]
                score = min(len(output) / 500, 1.0)
                return NumericResponse(
                    value=score,
                    reason=f"Output length: {len(output)} chars",
                )
        ```

        Use in evaluation:

        ```python
        results = evaluation.run(
            examples=examples,
            scorers=[ToxicityJudge(), LengthScorer()],
            eval_run_name="custom-eval",
        )
        ```
    """

    @abstractmethod
    async def score(self, data: Example) -> R:
        """Evaluate a single example and return a score.

        Implement this method with your scoring logic. Access example
        properties via bracket notation (e.g. `data["input"]`,
        `data["actual_output"]`).

        Args:
            data: The `Example` to evaluate.

        Returns:
            A response object (`BinaryResponse`, `NumericResponse`, or
            your `CategoricalResponse` subclass) with the score and reason.
        """
        pass
