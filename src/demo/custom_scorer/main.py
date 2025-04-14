from scorer import CustomScorer
from judgeval import JudgmentClient
from judgeval.data import Example, CustomExample
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer

client = JudgmentClient()

example = CustomExample(
    input={
        "question": "What if these shoes don't fit?",
    },
    actual_output={
        "answer": "We offer a 30-day full refund at no extra cost.",
    },
    expected_output={
        "answer": "We offer a 30-day full refund at no extra cost.",
    },
)

example2 = CustomExample(
    input={
        "question": "What if these shoes don't fit?",
    },
    actual_output={
        "answer": "We offer a 30-day full refund at no extra cost.",
    },
    expected_output={
        "answer": "We offer a 30-day full refund at no extra cost.",
    },
)

scorer = CustomScorer(threshold=0.5)
scorer2 = AnswerRelevancyScorer(threshold=0.5)
results = client.run_evaluation(
    project_name="custom-scorer-demo",
    eval_run_name="test-run4",
    examples=[example],
    scorers=[scorer],
    model="gpt-4o-mini",
    override=True,
)
print(results)

