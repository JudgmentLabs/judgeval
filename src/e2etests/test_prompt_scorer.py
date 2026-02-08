from judgeval.v1.scorers.prompt_scorer.prompt_scorer import PromptScorer
from uuid import uuid4
from judgeval.v1 import Judgeval
from judgeval.v1.data import Example
from e2etests.utils import retrieve_score
import time
from e2etests.utils import create_project, delete_project
from judgeval.v1.data.scoring_result import ScoringResult

QUERY_RETRY = 60


def _create_prompt_scorer(
    client: Judgeval, name: str, prompt: str, options=None, is_trace=False
):
    payload = {
        "name": name,
        "prompt": prompt,
        "threshold": 0.5,
        "model": "gpt-4o-mini",
        "is_trace": is_trace,
    }
    if options:
        payload["options"] = options
    client._internal_client.post_projects_scorers(
        project_id=client._project_id,
        payload=payload,
    )
    return PromptScorer(
        name=name,
        prompt=prompt,
        threshold=0.5,
        options=options,
        model="gpt-4o-mini",
        is_trace=is_trace,
        project_id=client._project_id,
    )


def test_prompt_scorer_without_options(client: Judgeval):
    scorer_name = f"Test Prompt Scorer Without Options {uuid4()}"
    prompt_scorer = _create_prompt_scorer(
        client,
        name=scorer_name,
        prompt="Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response relevant to the question?",
    )

    relevant_example = Example.create(
        input="What's the weather in New York?",
        actual_output="The weather in New York is sunny.",
    )

    irrelevant_example = Example.create(
        input="What's the capital of France?",
        actual_output="The mitochondria is the powerhouse of the cell, and did you know that honey never spoils?",
    )

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=[relevant_example, irrelevant_example],
        scorers=[prompt_scorer],
        eval_run_name="test-run-prompt-scorer-without-options",
    )

    assert res[0].success, "Relevant example should pass classification"
    assert not res[1].success, "Irrelevant example should fail classification"

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])


def test_prompt_scorer_with_options(client: Judgeval):
    scorer_name = f"Test Prompt Scorer {uuid4()}"
    prompt_scorer = _create_prompt_scorer(
        client,
        name=scorer_name,
        prompt="Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?",
        options={"yes": 1.0, "no": 0.0},
    )

    helpful_example = Example.create(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
    )

    unhelpful_example = Example.create(
        input="What's the capital of France?",
        actual_output="I don't know much about geography, but I think it might be somewhere in Europe.",
    )

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=[helpful_example, unhelpful_example],
        scorers=[prompt_scorer],
        eval_run_name="test-run-prompt-scorer-with-options",
    )

    assert res[0].success, "Helpful example should pass classification"
    assert not res[1].success, "Unhelpful example should fail classification"

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])


def test_get_prompt_scorer(client: Judgeval):
    random_id = uuid4()
    scorer_name = f"Test Prompt Scorer {random_id}"
    _create_prompt_scorer(
        client,
        name=scorer_name,
        prompt="Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?",
        options={"yes": 1.0, "no": 0.0},
    )
    prompt_scorer = client.scorers.prompt_scorer.get(name=scorer_name)
    assert prompt_scorer is not None
    assert prompt_scorer.get_name() == scorer_name
    assert (
        prompt_scorer.get_prompt()
        == "Question: {{input}}\nResponse: {{actual_output}}\n\nIs this response helpful?"
    )
    assert prompt_scorer.get_options() == {"yes": 1.0, "no": 0.0}


def test_custom_prompt_scorer(client: Judgeval):
    scorer_name = f"Test Prompt Scorer {uuid4()}"
    prompt_scorer = _create_prompt_scorer(
        client,
        name=scorer_name,
        prompt="Comparison A: {{comparison_a}}\n Comparison B: {{comparison_b}}\n\n Which candidate is better for a teammate?",
        options={"comparison_a": 1.0, "comparison_b": 0.0},
    )

    example1 = Example.create(
        comparison_a="Mike loves to play basketball because he passes with his teammates.",
        comparison_b="Mike likes to play 1v1 basketball because he likes to show off his skills.",
    )

    example2 = Example.create(
        comparison_a="Mike loves to play singles tennis because he likes to only hit by himself and not with a partner and is selfish.",
        comparison_b="Mike likes to play doubles tennis because he likes to coordinate with his partner.",
    )

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=[example1, example2],
        scorers=[prompt_scorer],
        eval_run_name="test-custom-prompt-scorer",
    )

    assert res[0].success, "Example 1 should pass classification"
    assert not res[1].success, "Example 2 should fail classification"

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])


def test_trace_prompt_scorer():
    project_name = f"test-trace-prompt-scorer-{uuid4()}"
    delete_project(project_name=project_name)
    create_project(project_name=project_name)
    judgeval_client = Judgeval(project_name=project_name)
    judgment = judgeval_client.tracer.create()

    trace_scorer = _create_prompt_scorer(
        judgeval_client,
        name=f"Test Trace Prompt Scorer {uuid4()}",
        prompt="Does this trace seem to represent a sample/test trace used for testing?",
        is_trace=True,
    )

    @judgment.observe(span_type="function")
    def sample_trace_span(sample_arg):
        print(f"This is a sample trace span with sample arg {sample_arg}")

    @judgment.observe(span_type="function")
    def main():
        sample_trace_span("test")
        judgment.async_trace_evaluate(scorer=trace_scorer)
        return (
            format(judgment.get_current_span().get_span_context().trace_id, "032x"),
            format(judgment.get_current_span().get_span_context().span_id, "016x"),
        )

    trace_id, span_id = main()
    query_count = 0
    scorer_data = None
    while query_count < QUERY_RETRY:
        scorer_data = retrieve_score(project_name, span_id, trace_id)
        if scorer_data:
            break
        query_count += 1
        time.sleep(1)
    delete_project(project_name=project_name)
    assert scorer_data[0].get("scorer_success")


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
