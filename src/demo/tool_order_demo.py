from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer
client = JudgmentClient()

if __name__ == "__main__":
    client.run_evaluation(
        examples=[
            Example(
                input="I want to plan a trip to Paris.",
                actual_output="I want to plan a trip to Paris.",
                expected_tools=[
                    {"agent": "TravelAgent", "tool": "get_attractions", "params": {"city": "Paris"}},
                    {"agent": "TravelAgent", "tool": "get_weather", "params": {"city": "Paris"}},
                ]
            )
        ],
        scorers=[
            FaithfulnessScorer(threshold=0.5)
        ],
        model="gpt-4o-mini",
        eval_run_name="tool_order_demo",
        project_name="tool_order_demo",
    )


