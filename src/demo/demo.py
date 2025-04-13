from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import DerailmentScorer

client = JudgmentClient()

example = Example(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

example2 = Example(
    input="What if I dont like the product?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

sequence = Sequence(
    name="Refund Policy",
    items=[example, example2],
)


scorer = DerailmentScorer(threshold=0.5)
results = client.run_sequence_evaluation(
    eval_run_name="test-sequence-run",
    project_name="test-sequence-project",
    sequence=sequence,
    scorers=[scorer],
    model="gpt-4o",
    log_results=True,
    override=True,
)
print(results)