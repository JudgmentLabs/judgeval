from judgeval import Judgeval
from judgeval.data import Example
from judgeval.judges import Judge
from judgeval.hosted.responses import BinaryResponse, NumericResponse


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


class OutputLength(Judge[NumericResponse]):
    """Scores output based on whether it stays within a reasonable length (under 300 chars)."""

    async def score(self, data: Example) -> NumericResponse:
        length = len(data["actual_output"])
        score = max(1.0 - (length / 300), 0.0)
        return NumericResponse(
            value=round(score, 2),
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
    assert_test=True,
)

for result in results:
    print(f"Success: {result.success}")
    for scorer in result.scorers_data:
        print(f"  {scorer.name}: score={scorer.score}, reason={scorer.reason}")
