from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import DerailmentScorer

client = JudgmentClient()

example = Example(
    input="How do I prepare this recipe?",
    actual_output="Here are the steps: Preheat the oven, mix the ingredients, bake for 30 minutes, etc.",
)
example2 = Example(
    input="What is the weather like?",
    actual_output="It's sunny with a high of 75°F."
)
example3 = Example(
    input="What is recipe step 5 again?",
    actual_output="Recipe step 5: Let the dough rest for 10 minutes"
)

nested_example = Example(
    input="What is recipe step 5 again?",
    actual_output="Recipe step 5: Let the dough rest for 10 minutes",
)   

nested_example2 = Example(
    input="What is recipe step 5 again?",
    actual_output="Recipe step 5: Let the dough rest for 10 minutes",
)

nested_sequence = Sequence(
    name="Recipe Steps",
    items=[nested_example, nested_example2],
    scorers=[DerailmentScorer(threshold=0.5)]
)

sequence = Sequence(
    name="Refund Policy",
    items=[example, example2, nested_sequence, example3],
    scorers=[DerailmentScorer(threshold=0.5)]
)

scorer = DerailmentScorer(threshold=0.5)
results = client.run_sequence_evaluation(
    eval_run_name="test-sequence-run",
    project_name="test-sequence-project",
    sequence=sequence,
    model="gpt-4o",
    log_results=True,
    override=True,
)
print(results)