from judgeval.scorers import PromptScorer, TracePromptScorer
from uuid import uuid4
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.tracer import Tracer, TraceScorerConfig
from e2etests.utils import retrieve_score
import time
from e2etests.utils import create_project, delete_project

QUERY_RETRY = 60


def test_prompt_scorer_without_options(client: JudgmentClient, project_name: str):
    """Test prompt scorer functionality."""

    prompt_scorer = PromptScorer.create(
        name=f"Test Prompt Scorer Without Options {uuid4()}",
        prompt="Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response relevant to the question?",
    )

    relevant_example = Example(
        input="What's the weather in New York?",
        actual_output="The weather in New York is sunny.",
    )

    irrelevant_example = Example(
        input="What's the capital of France?",
        actual_output="The mitochondria is the powerhouse of the cell, and did you know that honey never spoils?",
    )

    # Run evaluation
    res = client.run_evaluation(
        examples=[relevant_example, irrelevant_example],
        scorers=[prompt_scorer],
        project_name=project_name,
        eval_run_name="test-run-prompt-scorer-without-options",
    )

    # Verify results
    assert res[0].success, "Relevant example should pass classification"
    assert not res[1].success, "Irrelevant example should fail classification"

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])


def test_prompt_scorer_with_options(client: JudgmentClient, project_name: str):
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
    helpful_example = Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
    )

    unhelpful_example = Example(
        input="What's the capital of France?",
        actual_output="I don't know much about geography, but I think it might be somewhere in Europe.",
    )

    # Run evaluation
    res = client.run_evaluation(
        examples=[helpful_example, unhelpful_example],
        scorers=[prompt_scorer],
        project_name=project_name,
        eval_run_name="test-run-prompt-scorer-with-options",
    )

    # Verify results
    assert res[0].success, "Helpful example should pass classification"
    assert not res[1].success, "Unhelpful example should fail classification"

    # Print debug info if any test fails
    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])


def test_get_and_edit_prompt_scorer(client: JudgmentClient, project_name: str):
    random_id = uuid4()
    PromptScorer.create(
        name=f"Test Prompt Scorer {random_id}",
        prompt="Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?",
        options={"yes": 1.0, "no": 0.0},
    )
    prompt_scorer = PromptScorer.get(
        name=f"Test Prompt Scorer {random_id}",
    )
    assert prompt_scorer is not None
    assert prompt_scorer.name == f"Test Prompt Scorer {random_id}"
    assert (
        prompt_scorer.prompt
        == "Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?"
    )

    assert prompt_scorer.options == {"yes": 1.0, "no": 0.0}

    prompt_scorer.set_options({"y": 1.0, "n": 0.0})
    prompt_scorer.set_threshold(0.8)

    prompt_scorer2 = PromptScorer.get(
        name=f"Test Prompt Scorer {random_id}",
    )
    assert prompt_scorer2 is not None
    assert prompt_scorer2.name == f"Test Prompt Scorer {random_id}"
    assert (
        prompt_scorer2.prompt
        == "Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?"
    )
    assert prompt_scorer2.options == {"y": 1.0, "n": 0.0}
    assert prompt_scorer2.threshold == 0.8


def test_custom_prompt_scorer(client: JudgmentClient, project_name: str):
    """Test custom prompt scorer functionality."""
    # Creating a custom prompt scorer from SDK
    # Creating a prompt scorer from SDK
    prompt_scorer = PromptScorer.create(
        name=f"Test Prompt Scorer {uuid4()}",
        prompt="Comparison A: {{comparison_a}}\n Comparison B: {{comparison_b}}\n\n Which candidate is better for a teammate?",
        options={"comparison_a": 1.0, "comparison_b": 0.0},
    )

    prompt_scorer.set_options(
        {
            "comparison_a": 1.0,
            "comparison_b": 0.0,
        }
    )

    class ComparisonExample(Example):
        comparison_a: str
        comparison_b: str

    # Create test examples
    example1 = ComparisonExample(
        comparison_a="Mike loves to play basketball because he passes with his teammates.",
        comparison_b="Mike likes to play 1v1 basketball because he likes to show off his skills.",
    )

    example2 = ComparisonExample(
        comparison_a="Mike loves to play singles tennis because he likes to only hit by himself and not with a partner and is selfish.",
        comparison_b="Mike likes to play doubles tennis because he likes to coordinate with his partner.",
    )

    # Run evaluation
    res = client.run_evaluation(
        examples=[example1, example2],
        scorers=[prompt_scorer],
        project_name=project_name,
        eval_run_name="test-custom-prompt-scorer",
    )
    print(res)

    # Verify results
    assert res[0].success, "Example 1 should pass classification"
    assert not res[1].success, "Example 2 should fail classification"

    # Print debug info if any test fails
    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])


def test_trace_prompt_scorer():
    """Test trace prompt scorer functionality."""
    project_name = f"test-trace-prompt-scorer-{uuid4()}"
    delete_project(project_name=project_name)
    create_project(project_name=project_name)
    judgment = Tracer(project_name=project_name)
    trace_scorer = TracePromptScorer.create(
        name=f"Test Trace Prompt Scorer {uuid4()}", prompt="sample prompt"
    )
    trace_scorer.set_threshold(0.5)
    trace_scorer.set_prompt(
        "Does this trace seem to represent a sample/test trace used for testing?"
    )

    @judgment.observe(span_type="function")
    def sample_trace_span(sample_arg):
        print(f"This is a sample trace span with sample arg {sample_arg}")

    @judgment.observe(
        span_type="function",
        scorer_config=TraceScorerConfig(scorer=trace_scorer, model="gpt-4o-mini"),
    )
    def main():
        sample_trace_span("test")
        return (
            format(judgment.get_current_span().get_span_context().trace_id, "032x"),
            format(judgment.get_current_span().get_span_context().span_id, "016x"),
        )

    trace_id, span_id = main()
    query_count = 0
    while query_count < QUERY_RETRY:
        scorer_data = retrieve_score(span_id, trace_id)
        if scorer_data:
            break
        query_count += 1
        time.sleep(1)
    delete_project(project_name=project_name)
    assert scorer_data[0].get("scorer_success")


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
