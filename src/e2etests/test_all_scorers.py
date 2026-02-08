from judgeval.v1 import Judgeval
from judgeval.v1.scorers.built_in import (
    AnswerCorrectnessScorer,
    AnswerRelevancyScorer,
    FaithfulnessScorer,
)
from judgeval.v1.data import Example
from judgeval.v1.data.scoring_result import ScoringResult


def test_ac_scorer(client: Judgeval, random_name: str):
    example = Example.create(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
    )

    scorer = AnswerCorrectnessScorer(threshold=0.5)

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=[example],
        scorers=[scorer],
        eval_run_name=random_name,
    )
    print_debug_on_failure(res[0])


def test_ar_scorer(client: Judgeval, random_name: str):
    example_1 = Example.create(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
    )

    example_2 = Example.create(
        input="What's the capital of France?",
        actual_output="There's alot to do in Marseille. Lots of bars, restaurants, and museums.",
    )

    scorer = AnswerRelevancyScorer(threshold=0.5)

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=[example_1, example_2],
        scorers=[scorer],
        eval_run_name=random_name,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])

    assert res[0].success
    assert not res[1].success


def test_faithfulness_scorer(client: Judgeval, random_name: str):
    faithful_example = Example.create(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000.",
        ],
    )

    contradictory_example = Example.create(
        input="What's the capital of France?",
        actual_output="The capital of France is Lyon. It's located in southern France near the Mediterranean coast.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000.",
        ],
    )

    scorer = FaithfulnessScorer(threshold=1.0)

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=[faithful_example, contradictory_example],
        scorers=[scorer],
        eval_run_name=random_name,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])

    assert res[0].success
    assert not res[1].success, res[1]


def print_debug_on_failure(result: ScoringResult) -> bool:
    if not result.success:
        print(result.data_object)
        print("\nScorer Details:")
        for scorer_data in result.scorers_data:
            print(f"- Name: {scorer_data.name}")
            print(f"- Score: {scorer_data.score}")
            print(f"- Threshold: {scorer_data.threshold}")
            print(f"- Success: {scorer_data.success}")
            print(f"- Reason: {scorer_data.reason}")
            print(f"- Error: {scorer_data.error}")

        return False
    return True
