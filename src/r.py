from judgeval import Judgeval
from judgeval.v1.data import Example

judgeval = Judgeval(project_name="test")

Scorer = judgeval.scorers.prompt_scorer.get("Calculator")
assert Scorer is not None

evaluation = judgeval.evaluation.create()
evaluation.run(
    examples=[
        Example.create(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
        )
    ],
    scorers=[Scorer],
    eval_run_name="test",
)
