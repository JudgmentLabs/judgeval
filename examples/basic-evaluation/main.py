from typing import ClassVar, List

from judgeval import Judgeval
from judgeval.data import Example
from judgeval.judges import Judge
from judgeval.hosted.responses import (
    BinaryResponse,
    CategoricalResponse,
    Category,
)


class ContainsExpectedAnswer(Judge[BinaryResponse]):
    """Checks that the expected answer appears in the actual output."""

    async def score(self, data: Example) -> BinaryResponse:
        expected = data["expected_output"].strip().lower()
        actual = data["actual_output"].strip().lower()
        passed = expected in actual
        return BinaryResponse(
            value=passed,
            reason=f"Expected answer {'found' if passed else 'not found'} in output",
        )


class LengthCategory(CategoricalResponse):
    categories: ClassVar[List[Category]] = [
        Category(value="Concise", description="The output is under 300 characters."),
        Category(value="Long", description="The output is 300 characters or longer."),
    ]


class OutputLength(Judge[LengthCategory]):
    """Classifies output based on whether it stays within a reasonable length."""

    async def score(self, data: Example) -> LengthCategory:
        length = len(data["actual_output"])
        return LengthCategory(
            value="Concise" if length < 300 else "Long",
            reason=f"Output is {length} characters",
        )


client = Judgeval(project_name="my-project")

examples = [
    Example.create(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
    ),
    Example.create(
        input="What is 12 * 8?",
        actual_output="12 * 8 = 96",
        expected_output="97",
    ),
]

evaluation = client.evaluation.create()
results = evaluation.run(
    examples=examples,
    scorers=[ContainsExpectedAnswer(), OutputLength()],
    eval_run_name="basic-eval",
)

for result in results:
    for scorer in result.scorers_data:
        print(f"  {scorer.name}: {scorer.value}")
