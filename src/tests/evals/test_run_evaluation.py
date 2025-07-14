import pytest
from unittest.mock import Mock, patch

from judgeval.run_evaluation import (
    send_to_rabbitmq,
    execute_api_eval,
    merge_results,
    check_missing_scorer_data,
    check_experiment_type,
    check_eval_run_name_exists,
    log_evaluation_results,
    run_with_spinner,
    check_examples,
    get_evaluation_status,
    _poll_evaluation_until_complete,
    await_with_spinner,
    SpinnerWrappedTask,
    assert_test,
)
from judgeval.data import Example, ScoringResult, ScorerData, Trace
from judgeval.evaluation_run import EvaluationRun
from judgeval.data.trace_run import TraceRun
from judgeval.scorers import FaithfulnessScorer
from judgeval.common.exceptions import JudgmentAPIError

# Mock data for testing
MOCK_API_KEY = "test_api_key"
MOCK_ORG_ID = "test_org_id"
MOCK_PROJECT_NAME = "test_project"
MOCK_EVAL_NAME = "test_eval"


@pytest.fixture
def mock_evaluation_run():
    return EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[FaithfulnessScorer(threshold=0.5)],
        project_name=MOCK_PROJECT_NAME,
        eval_name=MOCK_EVAL_NAME,
        judgment_api_key=MOCK_API_KEY,
        organization_id=MOCK_ORG_ID,
    )


@pytest.fixture
def mock_trace_run():
    return TraceRun(
        traces=[
            Trace(
                trace_id="test_trace_id",
                name="test_trace",
                created_at="2024-03-20T12:00:00Z",
                duration=1.0,
                trace_spans=[],
            )
        ],
        scorers=[FaithfulnessScorer(threshold=0.5)],
        project_name=MOCK_PROJECT_NAME,
        eval_name=MOCK_EVAL_NAME,
        judgment_api_key=MOCK_API_KEY,
        organization_id=MOCK_ORG_ID,
    )


@pytest.fixture
def mock_scoring_results():
    return [
        ScoringResult(
            success=True,
            scorers_data=[
                ScorerData(
                    name="test_scorer",
                    threshold=0.5,
                    success=True,
                    score=0.8,
                    reason="Test reason",
                    strict_mode=True,
                    evaluation_model="gpt-4",
                    error=None,
                    additional_metadata={"test": "metadata"},
                )
            ],
            data_object=Example(input="test", actual_output="test"),
        )
    ]


class TestRunEvaluation:
    @patch("judgeval.run_evaluation.requests.post")
    def test_send_to_rabbitmq(self, mock_post, mock_evaluation_run):
        mock_post.return_value.json.return_value = {"status": "success"}
        mock_post.return_value.ok = True

        result = send_to_rabbitmq(mock_evaluation_run)

        assert result == {"status": "success"}
        mock_post.assert_called_once()

    @patch("judgeval.run_evaluation.requests.post")
    def test_execute_api_eval_success(self, mock_post, mock_evaluation_run):
        mock_post.return_value.json.return_value = {"results": [{"success": True}]}
        mock_post.return_value.ok = True

        result = execute_api_eval(mock_evaluation_run)

        assert result == {"results": [{"success": True}]}
        mock_post.assert_called_once()

    @patch("judgeval.run_evaluation.requests.post")
    def test_execute_api_eval_failure(self, mock_post, mock_evaluation_run):
        mock_post.return_value.ok = False
        mock_post.return_value.json.return_value = {"detail": "Error message"}

        with pytest.raises(JudgmentAPIError):
            execute_api_eval(mock_evaluation_run)

    def test_merge_results(self, mock_scoring_results):
        api_results = mock_scoring_results
        local_results = mock_scoring_results

        merged = merge_results(api_results, local_results)

        assert len(merged) == len(api_results)
        assert (
            merged[0].scorers_data
            == api_results[0].scorers_data + local_results[0].scorers_data
        )

    def test_check_missing_scorer_data(self, mock_scoring_results):
        results = mock_scoring_results
        results[0].scorers_data = None

        checked_results = check_missing_scorer_data(results)

        assert checked_results == results

    @patch("judgeval.run_evaluation.requests.post")
    def test_check_experiment_type_success(self, mock_post):
        mock_post.return_value.ok = True

        check_experiment_type(
            MOCK_EVAL_NAME, MOCK_PROJECT_NAME, MOCK_API_KEY, MOCK_ORG_ID, False
        )

        mock_post.assert_called_once()

    @patch("judgeval.run_evaluation.requests.post")
    def test_check_experiment_type_failure(self, mock_post):
        mock_post.return_value.ok = False
        mock_post.return_value.json.return_value = {"detail": "Error message"}

        with pytest.raises(JudgmentAPIError):
            check_experiment_type(
                MOCK_EVAL_NAME, MOCK_PROJECT_NAME, MOCK_API_KEY, MOCK_ORG_ID, False
            )

    @patch("judgeval.run_evaluation.requests.post")
    def test_check_eval_run_name_exists_success(self, mock_post):
        mock_post.return_value.ok = True

        check_eval_run_name_exists(
            MOCK_EVAL_NAME, MOCK_PROJECT_NAME, MOCK_API_KEY, MOCK_ORG_ID
        )

        mock_post.assert_called_once()

    @patch("judgeval.run_evaluation.requests.post")
    def test_check_eval_run_name_exists_conflict(self, mock_post):
        mock_post.return_value.status_code = 409

        with pytest.raises(ValueError):
            check_eval_run_name_exists(
                MOCK_EVAL_NAME, MOCK_PROJECT_NAME, MOCK_API_KEY, MOCK_ORG_ID
            )

    @patch("judgeval.run_evaluation.requests.post")
    def test_log_evaluation_results_success(
        self, mock_post, mock_scoring_results, mock_evaluation_run
    ):
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = {"ui_results_url": "http://test.com"}

        result = log_evaluation_results(mock_scoring_results, mock_evaluation_run)

        assert "View Results" in result
        mock_post.assert_called_once()

    def test_run_with_spinner(self):
        def test_func():
            return "test_result"

        result = run_with_spinner("Testing: ", test_func)

        assert result == "test_result"

    def test_check_examples(self):
        examples = [Example(input="test", actual_output="test")]
        scorers = [FaithfulnessScorer(threshold=0.5)]

        # Mock input to simulate user entering 'y'
        with patch("builtins.input", return_value="y"):
            check_examples(examples, scorers)

    @pytest.mark.asyncio
    async def test_get_evaluation_status(self):
        with patch("judgeval.run_evaluation.requests.get") as mock_get:
            mock_get.return_value.ok = True
            mock_get.return_value.json.return_value = {"status": "completed"}

            result = await get_evaluation_status(
                MOCK_EVAL_NAME, MOCK_PROJECT_NAME, MOCK_API_KEY, MOCK_ORG_ID
            )

            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_poll_evaluation_until_complete(self):
        with patch("judgeval.run_evaluation.requests.get") as mock_get:
            mock_get.side_effect = [
                Mock(ok=True, json=lambda: {"status": "pending"}),
                Mock(ok=True, json=lambda: {"status": "running"}),
                Mock(
                    ok=True,
                    json=lambda: {
                        "status": "completed",
                        "results": [{"success": True}],
                    },
                ),
            ]

            # Mock the results fetch endpoint
            with (
                patch("judgeval.run_evaluation.requests.post") as mock_post,
                patch("asyncio.sleep"),
            ):
                mock_post.return_value.ok = True
                mock_post.return_value.json.return_value = {
                    "examples": [
                        {
                            "example_id": "test_id",
                            "scorer_data": [
                                {
                                    "name": "test_scorer",
                                    "threshold": 0.5,
                                    "success": True,
                                    "score": 0.8,
                                    "reason": "Test reason",
                                    "strict_mode": True,
                                    "evaluation_model": "gpt-4",
                                    "error": None,
                                    "additional_metadata": {"test": "metadata"},
                                }
                            ],
                        }
                    ]
                }

                results = await _poll_evaluation_until_complete(
                    MOCK_EVAL_NAME,
                    MOCK_PROJECT_NAME,
                    MOCK_API_KEY,
                    MOCK_ORG_ID,
                    expected_scorer_count=1,
                    original_examples=[Example(input="test", actual_output="test")],
                )

                # Verify that we got the expected results
                assert len(results) == 1
                assert isinstance(results[0], ScoringResult)
                assert results[0].success is True

                # Verify that we made the expected number of status checks
                assert mock_get.call_count == 3

                # Verify that we made one call to fetch results
                mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_await_with_spinner(self):
        async def test_task():
            return "test_result"

        result = await await_with_spinner(test_task(), "Testing: ")

        assert result == "test_result"

    def test_spinner_wrapped_task(self):
        async def test_task():
            return "test_result", "pretty_str"

        task = SpinnerWrappedTask(test_task(), "Testing: ")

        # Test that the task is awaitable
        assert hasattr(task, "__await__")

    def test_assert_test_success(self, mock_scoring_results):
        # All tests pass
        assert_test(mock_scoring_results)

    def test_assert_test_failure(self):
        # Create a failing result
        failing_result = ScoringResult(
            success=False,
            scorers_data=[
                ScorerData(
                    name="test_scorer",
                    threshold=0.5,
                    success=False,
                    score=0.3,
                    reason="Test failure",
                    strict_mode=True,
                    evaluation_model="gpt-4",
                    error=None,
                    additional_metadata={"test": "metadata"},
                )
            ],
            data_object=Example(input="test", actual_output="test"),
        )

        with pytest.raises(AssertionError):
            assert_test([failing_result])
