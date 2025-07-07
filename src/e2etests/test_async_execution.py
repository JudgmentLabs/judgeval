import asyncio
import os
import pytest
import uuid
from typing import List

from judgeval.data import Example, ScoringResult
from judgeval.scorers import AnswerCorrectnessScorer, AnswerRelevancyScorer


# Skip these tests if API keys aren't set
pytestmark = pytest.mark.skipif(
    os.environ.get("JUDGMENT_API_KEY") is None
    or os.environ.get("JUDGMENT_ORG_ID") is None,
    reason="JUDGMENT_API_KEY and JUDGMENT_ORG_ID environment variables must be set to run e2e tests",
)


@pytest.fixture
def examples() -> List[Example]:
    """Return a list of examples for testing."""
    return [
        Example(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
        ),
        Example(
            input="What is the capital of Italy?",
            actual_output="Rome is the capital of Italy.",
            expected_output="Rome",
        ),
        Example(
            input="What is the capital of Germany?",
            actual_output="Berlin is the capital of Germany.",
            expected_output="Berlin",
        ),
    ]


@pytest.fixture
def tools_examples() -> List[Example]:
    """Return a list of examples with tools for testing."""
    return [
        Example(
            input="Find the weather in San Francisco",
            actual_output="The weather in San Francisco is sunny.",
            expected_output="The weather in San Francisco is sunny.",
            tools_called=["get_weather(location='San Francisco')"],
            expected_tools=[
                {
                    "tool_name": "get_weather",
                    "parameters": {"location": "San Francisco"},
                }
            ],
        ),
    ]


@pytest.mark.asyncio
async def test_async_evaluation_direct_await(client, examples, project_name):
    """Test direct awaiting of an async evaluation."""
    # Set up scorers
    scorers = [AnswerCorrectnessScorer(threshold=0.9)]

    # Run the async evaluation
    task = client.run_evaluation(
        examples=examples,
        scorers=scorers,
        model="gpt-4o-mini",
        project_name=project_name,
        eval_run_name="async-direct-await",
        async_execution=True,
    )

    # Await the results directly
    results = await task

    # Verify results
    assert isinstance(results, list)
    assert len(results) == len(examples)
    for result in results:
        assert isinstance(result, ScoringResult)
        assert result.success is True
        assert len(result.scorers_data) == len(scorers)


@pytest.mark.asyncio
async def test_async_evaluation_multiple_scorers(client, tools_examples, project_name):
    """Test async evaluation with multiple scorers."""
    # Set up multiple scorers
    scorers = [
        AnswerCorrectnessScorer(threshold=0.7),
        AnswerRelevancyScorer(threshold=0.7),
    ]

    # Run the async evaluation
    task = client.run_evaluation(
        examples=tools_examples,
        scorers=scorers,
        model="gpt-4.1",
        project_name=project_name,
        eval_run_name="async-multi-scorers",
        async_execution=True,
    )

    # Await the results
    results = await task

    # Verify results
    assert isinstance(results, list)
    assert len(results) == len(tools_examples)
    for result in results:
        assert isinstance(result, ScoringResult)
        assert result.success is True
        # Each result should have data from both scorers
        assert len(result.scorers_data) == len(scorers)


@pytest.mark.asyncio
async def test_async_evaluation_with_other_tasks(client, examples, project_name):
    """Test running other async tasks while an evaluation is in progress."""
    # Set up scorers
    scorers = [AnswerCorrectnessScorer(threshold=0.9)]

    # Start the evaluation
    eval_task = client.run_evaluation(
        examples=examples,
        scorers=scorers,
        model="gpt-4o-mini",
        project_name=project_name,
        eval_run_name="async-with-other-tasks",
        async_execution=True,
    )

    # Define a separate task that does other work
    async def do_other_work():
        results = []
        for i in range(3):
            await asyncio.sleep(0.5)  # Simulate work
            results.append(f"Work {i} completed")
        return results

    # Run both tasks concurrently
    other_task = asyncio.create_task(do_other_work())
    done, pending = await asyncio.wait(
        [eval_task, other_task], return_when=asyncio.ALL_COMPLETED
    )

    # Get results from both tasks
    eval_results = await eval_task
    other_results = await other_task

    # Verify evaluation results
    assert isinstance(eval_results, list)
    assert len(eval_results) == len(examples)

    # Verify other work results
    assert len(other_results) == 3
    assert all("Work" in result for result in other_results)


@pytest.mark.asyncio
async def test_pull_async_evaluation_results(client, examples, project_name):
    """Test pulling results of an async evaluation after it's completed."""
    # Set up a unique evaluation name
    eval_run_name = f"async-pull-{uuid.uuid4().hex[:8]}"

    # Set up scorers
    scorers = [AnswerCorrectnessScorer(threshold=0.9)]

    # Run the async evaluation
    task = client.run_evaluation(
        examples=examples,
        scorers=scorers,
        model="gpt-4o-mini",
        project_name=project_name,
        eval_run_name=eval_run_name,
        async_execution=True,
    )

    # Await the results
    await task

    # Pull the results using the client API
    pulled_results = client.pull_eval(project_name, eval_run_name)

    # Verify the pulled results
    assert isinstance(pulled_results, dict)
    assert "examples" in pulled_results
    examples_data = pulled_results["examples"]
    assert len(examples_data) == len(examples)

    # Check that each example has scorer data
    for example_data in examples_data:
        assert "scorer_data" in example_data
        assert len(example_data["scorer_data"]) > 0
