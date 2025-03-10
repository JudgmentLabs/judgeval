from demo import text2es_pipeline
from fail_format_response import fail_format_response_pipeline
from judgeval.data import Example
from judgeval import JudgmentClient
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer

import asyncio

def test():
    result = asyncio.run(text2es_pipeline())

    final_response = result.get("final_response", "")

    correctness_test_example = Example(
        input="Show me the connection status of user john.doe",
        actual_output=final_response,
        expected_output="John Doe is connected on WiFi.",
        retrieval_context=[str(result.get("query_results"))]
    )

    client = JudgmentClient()

    results = client.run_evaluation(
        examples=[correctness_test_example],
        scorers=[
            FaithfulnessScorer(threshold=1.0),
            AnswerRelevancyScorer(threshold=1.0),
            AnswerCorrectnessScorer(threshold=1.0)
        ],
        model="gpt-4o",
        eval_run_name="JNPR-Mist-UT-1",
        project_name="JNPR-Mist"
    )

    print(results)


def fail_test():
    result = asyncio.run(fail_format_response_pipeline())

    example = Example(
        input="Show me all disconnected access points",
        actual_output=result.get("final_response", ""),
        expected_output="""
        I found 3 disconnected access points:
        
        1. AP-Building-A-Floor1 (ap-001): Last seen on May 15, 2023, running firmware 6.1.2
        2. AP-Building-B-Floor2 (ap-002): Last seen on May 16, 2023, running firmware 6.1.2
        3. AP-Building-C-Floor1 (ap-003): Last seen on May 16, 2023, running firmware 6.0.9
        
        All of these access points are currently disconnected. Would you like more details about any specific access point?
        """,
        retrieval_context=[str(result.get("query_results"))]
    )

    client = JudgmentClient()

    results = client.run_evaluation(
        examples=[example],
        scorers=[
            AnswerCorrectnessScorer(threshold=1.0),
            FaithfulnessScorer(threshold=0.7),
        ],
        model="gpt-4o",
        eval_run_name="JNPR-Mist-UT-Fail",
        project_name="JNPR-Mist"
    )

    print(results)

if __name__ == "__main__":
    fail_test()
