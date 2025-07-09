import asyncio
import pytest
from unittest.mock import patch

from judgeval.data import Example, ScoringResult
from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import APIScorerConfig
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.constants import APIScorerType


@pytest.fixture
def examples():
    """Return a list of test examples."""
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
    ]


@pytest.fixture
def scorers():
    """Return a list of API scorers."""
    return [
        APIScorerConfig(
            name="Test Scorer",
            score_type=APIScorerType.ANSWER_CORRECTNESS,
            threshold=0.7,
        )
    ]


@pytest.fixture
def judgment_client():
    """Return a mocked JudgmentClient."""
    with patch(
        "judgeval.judgment_client.validate_api_key", return_value=(True, "valid")
    ):
        client = JudgmentClient(api_key="fake_key", organization_id="fake_org")
        return client


@pytest.mark.asyncio
async def test_async_execution_returns_task(judgment_client, examples, scorers):
    """Test that run_evaluation with async_execution=True returns a Task."""
    # Patch the run_eval function to return a Task
    with patch("judgeval.judgment_client.run_eval") as mock_run_eval:
        # Create a mock task
        mock_task = asyncio.create_task(asyncio.sleep(0))
        mock_run_eval.return_value = mock_task

        # Call run_evaluation with async_execution=True
        result = judgment_client.run_evaluation(
            examples=examples, scorers=scorers, async_execution=True
        )

        # Check that run_eval was called with async_execution=True
        mock_run_eval.assert_called_once()
        call_args = mock_run_eval.call_args[1]
        assert call_args["async_execution"] is True

        # Check that the result is a Task
        assert isinstance(result, asyncio.Task)


@pytest.mark.asyncio
async def test_async_execution_result_awaitable(judgment_client, examples, scorers):
    """Test that the Task returned by run_evaluation can be awaited and returns results."""
    # Create a mock ScoringResult to be returned by the task
    mock_result = [
        ScoringResult(
            success=True,
            scorers_data=[],
            data_object=examples[0],
            name=None,
            trace_id=None,
            run_duration=None,
        )
    ]

    # Create an async function that returns the mock result
    async def mock_async_workflow():
        await asyncio.sleep(0.1)  # Small delay
        return mock_result

    # Patch run_eval to return a real task with our mock workflow
    with patch("judgeval.judgment_client.run_eval") as mock_run_eval:
        mock_task = asyncio.create_task(mock_async_workflow())
        mock_run_eval.return_value = mock_task

        # Call run_evaluation with async_execution=True
        task = judgment_client.run_evaluation(
            examples=examples, scorers=scorers, async_execution=True
        )

        # Await the task and check the result
        result = await task
        assert result == mock_result


@pytest.mark.asyncio
async def test_async_execution_error_handling(judgment_client, examples, scorers):
    """Test that errors in async execution are properly propagated."""

    # Create an async function that raises an error
    async def mock_async_error():
        await asyncio.sleep(0.1)  # Small delay
        raise JudgmentAPIError("Test API error")

    # Patch run_eval to return a task that will raise an error
    with patch("judgeval.judgment_client.run_eval") as mock_run_eval:
        mock_task = asyncio.create_task(mock_async_error())
        mock_run_eval.return_value = mock_task

        # Call run_evaluation with async_execution=True
        task = judgment_client.run_evaluation(
            examples=examples, scorers=scorers, async_execution=True
        )

        # Await the task and check that the error is propagated
        with pytest.raises(JudgmentAPIError) as excinfo:
            await task

        assert "Test API error" in str(excinfo.value)
