from braintrust import Eval
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer
judgment_client = JudgmentClient()

def use_judgment_scorer(input, expected, output):
    print(f"input: {input}, expected: {expected}, output: {output}")
    example = Example(
        input=input,
        expected_output=expected,
        actual_output=output
        )

    scorer = AnswerRelevancyScorer(threshold=0.5)
    results = judgment_client.run_evaluation(
        project_name="Travel Agent",
        eval_run_name="Answer Relevancy",
        examples=[example],
        scorers=[scorer],
        override=True,
        model="gpt-4o",
    )
    print(results)
    return results[0].scorers_data[0].score

 
prompt = f"""
Create a structured travel itinerary for a trip to Honolulu from 2025-05-01 to 2025-05-07.
"""
Eval(
        "Travel Agent",
        data=lambda: [
            {
                "input": prompt,
                "expected": "A structured travel itinerary for a trip to Honolulu from 2025-05-01 to 2025-05-07.",
                "actual_output": "A structured travel itinerary for a trip to Honolulu from 2025-05-01 to 2025-05-07."
            }
        ],
        task=lambda input: "Hi" + input,
        scores=[use_judgment_scorer],
    )