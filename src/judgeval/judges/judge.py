from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from judgeval.data.example import Example
from judgeval.hosted.responses import (
    BaseResponse,
)

R = TypeVar("R", bound=BaseResponse)


class Judge(ABC, Generic[R]):
    """Base class for building Code judges.

    Subclass `Judge` and implement the `score` method to create your own
    scorer that runs locally. The type parameter `R` determines the response
    format:

    - `Judge[BinaryResponse]` -- pass/fail scoring
    - `Judge[NumericResponse]` -- numeric scoring (e.g. 0.0 to 1.0)
    - `Judge[CategoricalResponse]` -- classification into categories

    Code judges are passed to `Evaluation.run()` just like hosted scorers.

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

    recorded_metrics: dict[str, float]
    """Numbers recorded with `metric`; unset until the first call."""

    def metric(self, name: str, value: float) -> None:
        """Record a named number about the trace being scored.

        Call this from `score` to report measurements such as tool calls,
        cost or latency, separately from the score you return. Recording the
        same name again replaces its value. Test suite runs show each metric
        on the test case and summarize it across the run; other evaluations
        ignore them.

        Args:
            name: The metric's name, for example `"tool_calls"`.
            value: A finite number.

        Raises:
            ValueError: If `value` is not a finite number.
            TypeError: If `value` cannot be converted to a number.

        Examples:
            ```python
            class ToolUseJudge(Judge[BinaryResponse]):
                async def score(self, data: Example) -> BinaryResponse:
                    spans = data.trace.spans if data.trace else []
                    tool_spans = [s for s in spans if s["span_kind"] == "tool"]
                    self.metric("tool_calls", len(tool_spans))
                    return BinaryResponse(
                        value=len(tool_spans) <= 5,
                        reason=f"{len(tool_spans)} tool calls",
                    )
            ```
        """
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Metric {name!r} must be a finite number, got {value!r}")
        # Kept on the instance rather than set in __init__, which subclasses
        # may override without calling super().
        self.__dict__.setdefault("recorded_metrics", {})[name] = number

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
