"""
base e2e tests for all default judgeval scorers
"""

from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import (
    AnswerCorrectnessScorer,
    AnswerRelevancyScorer,
    FaithfulnessScorer,
    InstructionAdherenceScorer,
    ExecutionOrderScorer,
    PromptScorer,
)
from uuid import uuid4
from judgeval.data import JudgevalExample


def test_ac_scorer(client: JudgmentClient, project_name: str):
    example = JudgevalExample(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
    )

    scorer = AnswerCorrectnessScorer(threshold=0.5)
    EVAL_RUN_NAME = "test-run-ac"

    res = client.run_evaluation(
        examples=[example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        project_name=project_name,
        eval_run_name=EVAL_RUN_NAME,
        override=True,
    )
    print_debug_on_failure(res[0])


def test_ar_scorer(client: JudgmentClient, project_name: str):
    example_1 = JudgevalExample(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
    )

    example_2 = JudgevalExample(  # should fail
        input="What's the capital of France?",
        actual_output="There's alot to do in Marseille. Lots of bars, restaurants, and museums.",
    )

    scorer = AnswerRelevancyScorer(threshold=0.5)

    EVAL_RUN_NAME = "test-run-ar"

    res = client.run_evaluation(
        examples=[example_1, example_2],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        project_name=project_name,
        eval_run_name=EVAL_RUN_NAME,
        override=True,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])

    assert res[0].success
    assert not res[1].success


def test_faithfulness_scorer(client: JudgmentClient, project_name: str):
    faithful_example = JudgevalExample(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000.",
        ],
    )

    contradictory_example = JudgevalExample(  # should fail
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

    EVAL_RUN_NAME = "test-run-faithfulness"

    res = client.run_evaluation(
        examples=[faithful_example, contradictory_example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        project_name=project_name,
        eval_run_name=EVAL_RUN_NAME,
        override=True,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])

    assert res[0].success  # faithful_example should pass
    assert not res[1].success, res[1]  # contradictory_example should fail


def test_instruction_adherence_scorer(client: JudgmentClient, project_name: str):
    example_1 = JudgevalExample(
        input="write me a poem about cars and then turn it into a joke, but also what is 5 +5?",
        actual_output="Cars on the road, they zoom and they fly, Under the sun or a stormy sky. Engines roar, tires spin, A symphony of motion, let the race begin. Now for the joke: Why did the car break up with the bicycle. Because it was tired of being two-tired! And 5 + 5 is 10.",
    )

    scorer = InstructionAdherenceScorer(threshold=0.5)

    EVAL_RUN_NAME = "test-run-instruction-adherence"

    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        project_name=project_name,
        eval_run_name=EVAL_RUN_NAME,
        override=True,
    )

    print_debug_on_failure(res[0])

    assert res[0].success


def test_execution_order_scorer(client: JudgmentClient, project_name: str):
    EVAL_RUN_NAME = "test-run-execution-order"

    example = JudgevalExample(
        input="What is the weather in New York and the stock price of AAPL?",
        actual_output=[
            "weather_forecast",
            "stock_price",
            "translate_text",
            "news_headlines",
        ],
        expected_output=[
            "weather_forecast",
            "stock_price",
            "news_headlines",
            "translate_text",
        ],
    )

    res = client.run_evaluation(
        examples=[example],
        scorers=[ExecutionOrderScorer(threshold=1, should_consider_ordering=True)],
        model="gpt-4.1-mini",
        project_name=project_name,
        eval_run_name=EVAL_RUN_NAME,
        override=True,
    )

    assert not res[0].success


def test_prompt_scorer(client: JudgmentClient, project_name: str):
    """Test prompt scorer functionality."""
    # Creating a prompt scorer from SDK
    prompt_scorer = PromptScorer.create(
        name=f"Test Prompt Scorer {uuid4()}",
        prompt="Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?",
        options={"yes": 1.0, "no": 0.0},
    )

    # Update the options with helpfulness classification choices
    prompt_scorer.set_options(
        {
            "yes": 1.0,  # Helpful response
            "no": 0.0,  # Unhelpful response
        }
    )

    # Create test examples
    helpful_example = JudgevalExample(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris. It's one of the most populous cities in Europe and is known for landmarks like the Eiffel Tower and the Louvre Museum.",
    )

    unhelpful_example = JudgevalExample(
        input="What's the capital of France?",
        actual_output="I don't know much about geography, but I think it might be somewhere in Europe.",
    )

    # Run evaluation
    res = client.run_evaluation(
        examples=[helpful_example, unhelpful_example],
        scorers=[prompt_scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        project_name=project_name,
        eval_run_name="test-run-helpfulness",
        override=True,
    )

    # Verify results
    assert res[0].success, "Helpful example should pass classification"
    assert not res[1].success, "Unhelpful example should fail classification"

    # Print debug info if any test fails
    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])


def print_debug_on_failure(result) -> bool:
    """
    Helper function to print debug info only on test failure

    Returns:
        bool: True if the test passed, False if it failed
    """
    if not result.success:
        print(result.data_object.model_dump())
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
