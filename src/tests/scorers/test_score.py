import pytest
from unittest.mock import AsyncMock, Mock, patch
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio

from judgeval.scorers.score import (
    safe_a_score_example, 
    score_task, 
    score_with_indicator, 
    a_execute_scoring, 
    a_eval_examples_helper
)
from judgeval.scorers import JudgevalScorer
from judgeval.data import Example, ScoringResult, ScorerData
from judgeval.common.exceptions import MissingTestCaseParamsError


class MockJudgevalScorer(JudgevalScorer):
    def score_example(self, example, *args, **kwargs):
        pass

    async def a_score_example(self, example, *args, **kwargs):
        pass

    def success_check(self):
        return True


@pytest.fixture
def example():
    return Example(
        input="test input",
        actual_output="test output",
        example_id="test_id"
    )


@pytest.fixture
def basic_scorer():
    return MockJudgevalScorer(
        score_type="test_scorer",
        threshold=0.5
    )


@pytest.fixture
def scorers(basic_scorer):
    """Fixture providing a list of test scorers"""
    return [
        MockJudgevalScorer(score_type="test_scorer", threshold=0.5),
        MockJudgevalScorer(score_type="test_scorer", threshold=0.5)
    ]


@pytest.fixture
def progress():
    return Progress(
        SpinnerColumn(style="rgb(106,0,255)"),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )


@pytest.mark.asyncio
async def test_successful_scoring(example, basic_scorer):
    """Test basic successful scoring case"""
    basic_scorer.a_score_example = AsyncMock()
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    basic_scorer.a_score_example.assert_called_once_with(example, _show_indicator=False)
    assert basic_scorer.error is None
    assert not hasattr(basic_scorer, 'skipped') or not basic_scorer.skipped


@pytest.mark.asyncio
async def test_missing_params_with_skip(example, basic_scorer):
    """Test handling of MissingTestCaseParamsError when skip_on_missing_params is True"""
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    assert basic_scorer.skipped is True
    assert basic_scorer.error is None


@pytest.mark.asyncio
async def test_missing_params_with_ignore_errors(example, basic_scorer):
    """Test handling of MissingTestCaseParamsError when ignore_errors is True but not skipping"""
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=False
    )
    
    assert basic_scorer.error == "Missing required params"
    assert basic_scorer.success is False


@pytest.mark.asyncio
async def test_missing_params_raises_error(example, basic_scorer):
    """Test that MissingTestCaseParamsError is raised when appropriate"""
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with pytest.raises(MissingTestCaseParamsError):
        await safe_a_score_example(
            scorer=basic_scorer,
            example=example,
            ignore_errors=False,
            skip_on_missing_params=False
        )


@pytest.mark.asyncio
async def test_type_error_handling(example, basic_scorer):
    """Test handling of TypeError when _show_indicator is not accepted"""
    calls = []
    
    async def mock_score(*args, **kwargs):
        calls.append(kwargs)
        if '_show_indicator' in kwargs:
            raise TypeError("_show_indicator not accepted")
        return True
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    assert len(calls) == 2  # Should try twice - once with _show_indicator, once without
    assert '_show_indicator' in calls[0]  # First attempt includes _show_indicator
    assert '_show_indicator' not in calls[1]  # Second attempt doesn't include _show_indicator


@pytest.mark.asyncio
async def test_general_exception_with_ignore(example, basic_scorer):
    """Test handling of general exceptions when ignore_errors is True"""
    async def mock_score(*args, **kwargs):
        raise ValueError("Test error")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    assert basic_scorer.error == "Test error"
    assert basic_scorer.success is False


@pytest.mark.asyncio
async def test_general_exception_raises(example, basic_scorer):
    """Test that general exceptions are raised when ignore_errors is False"""
    async def mock_score(*args, **kwargs):
        raise ValueError("Test error")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with pytest.raises(ValueError):
        await safe_a_score_example(
            scorer=basic_scorer,
            example=example,
            ignore_errors=False,
            skip_on_missing_params=True
        )


@pytest.mark.asyncio
async def test_error_with_missing_params(example, basic_scorer):
    """Test handling of TypeError followed by MissingTestCaseParamsError"""
    calls = []
    
    async def mock_score(*args, **kwargs):
        calls.append(kwargs)
        if '_show_indicator' in kwargs:
            raise TypeError("_show_indicator not accepted")
        raise MissingTestCaseParamsError("Missing params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    assert basic_scorer.skipped is True
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_task_successful_scoring(example, basic_scorer, progress):
    """Test basic successful scoring case with progress tracking"""
    task_id = progress.add_task(description="Test Task", total=100)
    basic_scorer.a_score_example = AsyncMock()
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example
        )
    
    basic_scorer.a_score_example.assert_called_once_with(example, _show_indicator=False)
    assert progress.tasks[task_id].completed == 100
    assert "Completed" in progress.tasks[task_id].description


@pytest.mark.asyncio
async def test_task_missing_params_with_skip(example, basic_scorer, progress):
    """Test handling of MissingTestCaseParamsError when skip_on_missing_params is True"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example,
            skip_on_missing_params=True
        )
    
    assert basic_scorer.skipped is True
    assert not progress.tasks[task_id].completed  # Task should not be marked as complete


@pytest.mark.asyncio
async def test_task_missing_params_with_ignore_errors(example, basic_scorer, progress):
    """Test handling of MissingTestCaseParamsError when ignore_errors is True"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example,
            skip_on_missing_params=False,
            ignore_errors=True
        )
    
    assert basic_scorer.error == "Missing required params"
    assert basic_scorer.success is False
    assert progress.tasks[task_id].completed == 100
    assert "Failed" in progress.tasks[task_id].description


@pytest.mark.asyncio
async def test_task_missing_params_raises_error(example, basic_scorer, progress):
    """Test that MissingTestCaseParamsError is raised when appropriate"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with pytest.raises(MissingTestCaseParamsError):
        with progress:
            await score_task(
                task_id=task_id,
                progress=progress,
                scorer=basic_scorer,
                example=example,
                skip_on_missing_params=False,
                ignore_errors=False
            )


@pytest.mark.asyncio
async def test_task_type_error_handling(example, basic_scorer, progress):
    """Test handling of TypeError when _show_indicator is not accepted"""
    task_id = progress.add_task(description="Test Task", total=100)
    calls = []
    
    async def mock_score(*args, **kwargs):
        calls.append(kwargs)
        if '_show_indicator' in kwargs:
            raise TypeError("_show_indicator not accepted")
        return True
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example
        )
    
    assert len(calls) == 2  # Should try twice - once with _show_indicator, once without
    assert progress.tasks[task_id].completed == 100
    assert "Completed" in progress.tasks[task_id].description


@pytest.mark.asyncio
async def test_task_general_exception_with_ignore(example, basic_scorer, progress):
    """Test handling of general exceptions when ignore_errors is True"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise ValueError("Test error")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example,
            ignore_errors=True
        )
    
    assert basic_scorer.error == "Test error"
    assert basic_scorer.success is False
    assert progress.tasks[task_id].completed == 100
    assert "Failed" in progress.tasks[task_id].description


@pytest.mark.asyncio
async def test_task_general_exception_raises(example, basic_scorer, progress):
    """Test that general exceptions are raised when ignore_errors is False"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise ValueError("Test error")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with pytest.raises(ValueError):
        with progress:
            await score_task(
                task_id=task_id,
                progress=progress,
                scorer=basic_scorer,
                example=example,
                ignore_errors=False
            )


@pytest.mark.asyncio
async def test_task_progress_timing(example, basic_scorer, progress):
    """Test that timing information is correctly added to progress description"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        await asyncio.sleep(0.1)  # Simulate some work
        return True
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example
        )
    
    assert "(" in progress.tasks[task_id].description
    assert "s)" in progress.tasks[task_id].description  # Should show timing


@pytest.mark.asyncio
@patch('judgeval.scorers.score.safe_a_score_example')
@patch('judgeval.scorers.score.score_task')
async def test_score_with_indicator_no_show(mock_score_task, mock_safe_score, example, scorers):
    """Test scoring without showing the indicator"""
    mock_safe_score.return_value = AsyncMock()()

    await score_with_indicator(
        scorers=scorers,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False
    )

    assert mock_safe_score.call_count == 2  # Called once for each scorer
    assert mock_score_task.call_count == 0  # Should not be called when show_indicator is False

@pytest.mark.asyncio
@patch('judgeval.scorers.score.Progress')
@patch('judgeval.scorers.score.score_task')
@patch('judgeval.scorers.score.scorer_console_msg')
async def test_score_with_indicator_show(mock_console_msg, mock_score_task, mock_progress, example, scorers):
    """Test scoring with progress indicator"""
    mock_progress_instance = Mock()
    mock_progress.return_value.__enter__.return_value = mock_progress_instance
    mock_progress_instance.add_task.return_value = 1
    mock_score_task.return_value = AsyncMock()()
    mock_console_msg.return_value = "Test Progress Message"

    await score_with_indicator(
        scorers=scorers,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=True
    )

    assert mock_progress_instance.add_task.call_count == 2  # Called once for each scorer
    assert mock_score_task.call_count == 2  # Called once for each scorer

@pytest.mark.asyncio
async def test_score_with_indicator_error_handling(example, scorers):
    """Test error handling during scoring"""
    # Make first scorer raise an error
    async def mock_error(*args, **kwargs):
        raise ValueError("Test error")
    
    async def mock_success(*args, **kwargs):
        # Simulate successful scoring
        scorers[1].success = True
        return True
    
    scorers[0].a_score_example = AsyncMock(side_effect=mock_error)
    scorers[1].a_score_example = AsyncMock(side_effect=mock_success)

    await score_with_indicator(
        scorers=scorers,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False
    )
    
    assert scorers[0].error == "Test error"
    assert scorers[0].success is False
    assert scorers[1].error is None
    assert scorers[1].success is True

@pytest.mark.asyncio
async def test_score_with_indicator_missing_params(example, scorers):
    """Test handling of missing parameters"""
    async def mock_missing_params(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing params")
    
    # Set up mock for first scorer to raise error
    scorers[0].a_score_example = AsyncMock(side_effect=mock_missing_params)
    # Set up mock for second scorer to succeed
    scorers[1].a_score_example = AsyncMock(return_value=True)

    await score_with_indicator(
        scorers=scorers,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False
    )
    
    assert scorers[0].skipped is True
    assert not hasattr(scorers[1], 'skipped')  # Second scorer should not be skipped, so attribute shouldn't exist

@pytest.mark.asyncio
async def test_score_with_indicator_raises_error(example, scorers):
    """Test that errors are raised when ignore_errors is False"""
    async def mock_error(*args, **kwargs):
        raise ValueError("Test error")

    scorers[0].a_score_example = AsyncMock(side_effect=mock_error)

    with pytest.raises(ValueError):
        await score_with_indicator(
            scorers=scorers,
            example=example,
            ignore_errors=False,  # Errors should be raised
            skip_on_missing_params=True,
            show_indicator=False
        )

@pytest.mark.asyncio
@patch('judgeval.scorers.score.Progress')
async def test_score_with_indicator_empty_scorers(mock_progress, example):
    """Test handling of empty scorers list"""
    await score_with_indicator(
        scorers=[],
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False
    )

    mock_progress.assert_not_called()

@pytest.mark.asyncio
@patch('judgeval.scorers.score.Progress')
async def test_score_with_indicator_concurrent_execution(mock_progress, example, scorers):
    """Test that scorers are executed concurrently"""
    completed_order = []
    
    async def mock_delayed_score(*args, **kwargs):
        await asyncio.sleep(0.1)  # First scorer
        completed_order.append(1)

    async def mock_quick_score(*args, **kwargs):
        completed_order.append(2)  # Second scorer

    # Create two separate scorer instances instead of using the same one twice
    scorer1 = MockJudgevalScorer(score_type="test_scorer", threshold=0.5)
    scorer2 = MockJudgevalScorer(score_type="test_scorer", threshold=0.5)
    
    scorer1.a_score_example = AsyncMock(side_effect=mock_delayed_score)
    scorer2.a_score_example = AsyncMock(side_effect=mock_quick_score)

    await score_with_indicator(
        scorers=[scorer1, scorer2],  # Use the new separate instances
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False
    )

    # Second scorer should complete before first scorer due to delay
    assert completed_order == [2, 1]


@pytest.fixture
def mock_example():
    return Example(
        input="test input",
        actual_output="test output",
        example_id="test_id",
        timestamp="20241225_000004"
    )

@pytest.fixture
def mock_examples():
    return [
        Example(input=f"test input {i}", 
               actual_output=f"test output {i}", 
               example_id=f"test_id_{i}",
               timestamp="20241225_000004")
        for i in range(3)
    ]

@pytest.fixture
def mock_scorer():
    class MockScorer(JudgevalScorer):
        def __init__(self):
            self.success = None
            self.error = None
            self.skipped = False
            self.verbose_mode = False
            self._add_model = Mock()
        
    return MockScorer()

@pytest.fixture
def mock_scoring_result():
    return Mock(spec=ScoringResult)

# Tests
@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
async def test_basic_execution(mock_helper, mock_clone_scorers, mock_examples, mock_scorer, mock_scoring_result):
    """Test basic execution with single scorer and multiple examples"""
    # Setup mocks
    mock_clone_scorers.return_value = [mock_scorer]
    mock_helper.return_value = None
    
    results = await a_execute_scoring(
        examples=mock_examples,
        scorers=[mock_scorer],
        show_indicator=False
    )
    
    assert len(results) == len(mock_examples)
    assert mock_helper.call_count == len(mock_examples)
    assert mock_clone_scorers.call_count == len(mock_examples)

@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
async def test_empty_scorers(mock_helper, mock_clone_scorers, mock_examples):
    """Test execution with no scorers"""
    results = await a_execute_scoring(
        examples=mock_examples,
        scorers=[],
        show_indicator=False
    )
    
    assert len(results) == len(mock_examples)
    mock_helper.assert_not_called()
    mock_clone_scorers.assert_not_called()

@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
async def test_empty_examples(mock_helper, mock_clone_scorers, mock_scorer):
    """Test execution with no examples"""
    results = await a_execute_scoring(
        examples=[],
        scorers=[mock_scorer],
        show_indicator=False
    )
    
    assert len(results) == 0
    mock_helper.assert_not_called()
    mock_clone_scorers.assert_not_called()

@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
async def test_error_handling(mock_helper, mock_clone_scorers, mock_examples, mock_scorer):
    """Test error handling when helper raises exception"""
    mock_clone_scorers.return_value = [mock_scorer]
    mock_helper.side_effect = ValueError("Test error")
    
    # Test with ignore_errors=True
    results = await a_execute_scoring(
        examples=mock_examples,
        scorers=[mock_scorer],
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False,
        _use_bar_indicator=False
    )
    
    # Add assertions to verify error was handled
    assert len(results) == len(mock_examples)
    assert all(result is None for result in results)  # Results should be None when errors are ignored
    
    # Test with ignore_errors=False
    with pytest.raises(ValueError):
        await a_execute_scoring(
            examples=mock_examples,
            scorers=[mock_scorer],
            ignore_errors=False,
            skip_on_missing_params=True,
            show_indicator=False,
            _use_bar_indicator=False
        )

@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
async def test_max_concurrent_limit(mock_helper, mock_clone_scorers, mock_examples, mock_scorer):
    """Test concurrent execution limit"""
    mock_clone_scorers.return_value = [mock_scorer]
    
    async def delayed_execution(*args, **kwargs):
        await asyncio.sleep(0.1)
        return None
    
    mock_helper.side_effect = delayed_execution
    
    start_time = asyncio.get_event_loop().time()
    
    await a_execute_scoring(
        examples=mock_examples,
        scorers=[mock_scorer],
        max_concurrent=1,  # Force sequential execution
        show_indicator=False
    )
    
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    
    # Duration should be at least (num_examples * 0.1) seconds due to sequential execution
    assert duration >= len(mock_examples) * 0.1

@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
async def test_throttle_value(mock_helper, mock_clone_scorers, mock_examples, mock_scorer):
    """Test throttling between tasks"""
    mock_clone_scorers.return_value = [mock_scorer]
    start_time = asyncio.get_event_loop().time()
    
    await a_execute_scoring(
        examples=mock_examples,
        scorers=[mock_scorer],
        throttle_value=0.1,
        show_indicator=False
    )
    
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    
    # Duration should be at least (num_examples - 1) * throttle_value
    assert duration >= (len(mock_examples) - 1) * 0.1

@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
@patch('judgeval.scorers.score.tqdm_asyncio')
async def test_progress_indicator(mock_tqdm, mock_helper, mock_clone_scorers, mock_examples, mock_scorer):
    """Test progress indicator functionality"""
    mock_clone_scorers.return_value = [mock_scorer]
    
    await a_execute_scoring(
        examples=mock_examples,
        scorers=[mock_scorer],
        show_indicator=True,
        _use_bar_indicator=True
    )
    
    assert mock_tqdm.called
    mock_helper.assert_called()

@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
async def test_model_assignment(mock_helper, mock_clone_scorers, mock_examples, mock_scorer):
    """Test model assignment to scorers"""
    mock_clone_scorers.return_value = [mock_scorer]
    test_model = "test_model"
    
    await a_execute_scoring(
        examples=mock_examples,
        scorers=[mock_scorer],
        model=test_model,
        show_indicator=False
    )
    
    mock_scorer._add_model.assert_called_once_with(test_model)

@pytest.mark.asyncio
@patch('judgeval.scorers.score.clone_scorers')
@patch('judgeval.scorers.score.a_eval_examples_helper')
async def test_verbose_mode_setting(mock_helper, mock_clone_scorers, mock_examples, mock_scorer):
    """Test verbose mode is properly set on scorers"""
    mock_clone_scorers.return_value = [mock_scorer]
    
    await a_execute_scoring(
        examples=mock_examples,
        scorers=[mock_scorer],
        verbose_mode=True,
        show_indicator=False
    )
    
    assert mock_scorer.verbose_mode is True


@pytest.fixture
def mock_example():
    """Create a mock Example object"""
    return Example(
        name="test_example",
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context=["context1", "context2"],
        retrieval_context=["retrieval1"],
    )

@pytest.fixture
def mock_scorer():
    """Create a mock JudgevalScorer"""
    scorer = Mock(spec=JudgevalScorer)
    scorer.__name__ = "MockScorer"
    scorer.threshold = 0.8
    scorer.strict_mode = True
    scorer.evaluation_model = "test-model"
    scorer.score = 0.9
    scorer.reason = "Test reason"
    scorer.success_check.return_value = True
    scorer.evaluation_cost = 0.1
    scorer.verbose_logs = "Test logs"
    scorer.additional_metadata = {"key": "value"}
    scorer.skipped = False
    scorer.error = None
    return scorer

@pytest.fixture
def mock_scoring_results():
    """Create a mock list to store ScoringResults"""
    return [None] * 3  # List with 3 None elements

@pytest.mark.asyncio
async def test_a_eval_examples_helper_success(
    mock_example,
    mock_scorer,
    mock_scoring_results,
):
    """Test successful execution of a_eval_examples_helper"""
    
    # Create list of scorers
    scorers = [mock_scorer]
    
    # Mock the external functions
    with patch('judgeval.scorers.score.score_with_indicator', new_callable=AsyncMock) as mock_score_with_indicator, \
         patch('judgeval.scorers.score.create_scorer_data') as mock_create_scorer_data, \
         patch('judgeval.scorers.score.generate_scoring_result') as mock_generate_result:
        
        # Setup mock returns
        mock_scorer_data = ScorerData(
            name=mock_scorer.__name__,
            threshold=mock_scorer.threshold,
            success=True,
            score=mock_scorer.score,
            reason=mock_scorer.reason,
            strict_mode=mock_scorer.strict_mode,
            evaluation_model=mock_scorer.evaluation_model,
            error=None,
            evaluation_cost=mock_scorer.evaluation_cost,
            verbose_logs=mock_scorer.verbose_logs,
            additional_metadata=mock_scorer.additional_metadata
        )
        mock_create_scorer_data.return_value = mock_scorer_data
        
        mock_scoring_result = ScoringResult(
            success=True,
            scorers_data=[mock_scorer_data],
            data_object=mock_example,
            trace_id=mock_example.trace_id
        )
        mock_generate_result.return_value = mock_scoring_result

        # Execute the function
        await a_eval_examples_helper(
            scorers=scorers,
            example=mock_example,
            scoring_results=mock_scoring_results,
            score_index=0,
            ignore_errors=True,
            skip_on_missing_params=True,
            show_indicator=True,
            _use_bar_indicator=False,
            pbar=None
        )

        # Verify the calls
        mock_score_with_indicator.assert_called_once_with(
            scorers=scorers,
            example=mock_example,
            skip_on_missing_params=True,
            ignore_errors=True,
            show_indicator=True
        )
        mock_create_scorer_data.assert_called_once_with(mock_scorer)
        
        # Verify the result was stored correctly
        assert mock_scoring_results[0] == mock_scoring_result

@pytest.mark.asyncio
async def test_a_eval_examples_helper_with_skipped_scorer(
    mock_example,
    mock_scorer,
    mock_scoring_results,
):
    """Test execution when scorer is skipped"""
    
    scorers = [mock_scorer]
    
    with patch('judgeval.scorers.score.score_with_indicator', new_callable=AsyncMock) as mock_score_with_indicator, \
         patch('judgeval.scorers.score.create_scorer_data') as mock_create_scorer_data, \
         patch('judgeval.scorers.score.generate_scoring_result') as mock_generate_result:
        
        # Mock score_with_indicator to simulate skipped scorer behavior
        async def mock_score(*args, **kwargs):
            # Set scorer as skipped after score_with_indicator is called
            mock_scorer.skipped = True
            return None
            
        mock_score_with_indicator.side_effect = mock_score
        
        await a_eval_examples_helper(
            scorers=scorers,
            example=mock_example,
            scoring_results=mock_scoring_results,
            score_index=1,
            ignore_errors=True,
            skip_on_missing_params=True,
            show_indicator=True,
            _use_bar_indicator=False,
            pbar=None
        )

        # Verify that create_scorer_data was not called since scorer was skipped
        mock_create_scorer_data.assert_not_called()

@pytest.mark.asyncio
async def test_a_eval_examples_helper_with_progress_bar(
    mock_example,
    mock_scorer,
    mock_scoring_results,
):  
    """Test execution with progress bar"""
    
    scorers = [mock_scorer]
    mock_pbar = Mock()
    
    with patch('judgeval.scorers.score.score_with_indicator', new_callable=AsyncMock), \
         patch('judgeval.scorers.score.create_scorer_data'), \
         patch('judgeval.scorers.score.generate_scoring_result'):
        
        await a_eval_examples_helper(
            scorers=scorers,
            example=mock_example,
            scoring_results=mock_scoring_results,
            score_index=2,
            ignore_errors=True,
            skip_on_missing_params=True,
            show_indicator=True,
            _use_bar_indicator=True,
            pbar=mock_pbar
        )

        # Verify progress bar was updated
        mock_pbar.update.assert_called_once_with(1)

