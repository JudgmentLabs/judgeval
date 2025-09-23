from judgeval.scorers import PromptScorer, TracePromptScorer
from uuid import uuid4
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.env import JUDGMENT_DEFAULT_TOGETHER_MODEL
from judgeval.tracer import Tracer, TraceScorerConfig
from e2etests.utils import retrieve_score
import time
from e2etests.utils import create_project, delete_project

QUERY_RETRY = 60


def test_prompt_scorer(client: JudgmentClient, project_name: str):
    """Test prompt scorer functionality."""

    prompt_scorer = PromptScorer.create(
        name=f"Test Prompt Scorer {uuid4()}",
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
        model=JUDGMENT_DEFAULT_TOGETHER_MODEL,
        project_name=project_name,
        eval_run_name="test-run-prompt-scorer",
    )

    # Verify results
    assert res[0].success, "Relevant example should pass classification"
    assert not res[1].success, "Irrelevant example should fail classification"

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])


def test_get_and_edit_prompt_scorer(client: JudgmentClient, project_name: str):
    random_id = uuid4()
    PromptScorer.create(
        name=f"Test Prompt Scorer {random_id}",
        prompt="Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?",
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

    prompt_scorer.append_to_prompt(
        "Consider accuracy, clarity, and completeness in your evaluation."
    )
    prompt_scorer.set_threshold(0.8)

    prompt_scorer2 = PromptScorer.get(
        name=f"Test Prompt Scorer {random_id}",
    )
    assert prompt_scorer2 is not None
    assert prompt_scorer2.name == f"Test Prompt Scorer {random_id}"
    assert (
        prompt_scorer2.prompt
        == "Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?Consider accuracy, clarity, and completeness in your evaluation."
    )
    assert prompt_scorer2.threshold == 0.8


def test_trace_prompt_scorer(project_name: str):
    """Test trace prompt scorer functionality."""
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
        scorer_config=TraceScorerConfig(scorer=trace_scorer, model="gpt-5"),
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
    assert scorer_data[0].get("success")


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
