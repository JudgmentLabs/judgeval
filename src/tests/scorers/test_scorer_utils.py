import pytest
import asyncio

from judgeval.scorers.utils import (
    clone_scorers,
    scorer_console_msg,
    scorer_progress_meter,
    parse_response_json,
    print_verbose_logs,
    create_verbose_logs,
    get_or_create_event_loop,
)
from judgeval.scorers import BaseScorer
from judgeval.data import Example


class MockBaseScorer(BaseScorer):
    """Mock implementation of BaseScorer for testing"""

    def __init__(self, **kwargs):
        super().__init__(score_type="mock_scorer", threshold=0.7, **kwargs)
        self.__name__ = "MockScorer"

    def score_example(self, example: Example, *args, **kwargs) -> float:
        return 1.0

    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        return 1.0

    def success_check(self) -> bool:
        return True


@pytest.fixture
def mock_scorer():
    return MockBaseScorer(
        evaluation_model="gpt-4", strict_mode=True, async_mode=True, verbose_mode=True
    )


@pytest.fixture
def mock_scorers():
    return [
        MockBaseScorer(evaluation_model="gpt-4.1"),
        MockBaseScorer(evaluation_model="gpt-4.1"),
    ]


def test_clone_scorers(mock_scorers):
    """Test that scorers are properly cloned with all attributes"""
    cloned = clone_scorers(mock_scorers)

    assert len(cloned) == len(mock_scorers)
    for original, clone in zip(mock_scorers, cloned):
        assert type(original) is type(clone)
        assert original.score_type == clone.score_type
        assert original.threshold == clone.threshold
        assert original.evaluation_model == clone.evaluation_model


def test_scorer_console_msg(mock_scorer):
    """Test console message formatting"""
    # Test with default async_mode
    msg = scorer_console_msg(mock_scorer)
    assert "MockScorer" in msg
    assert "gpt-4" in msg
    assert "async_mode=True" in msg

    # Test with explicit async_mode
    msg = scorer_console_msg(mock_scorer, async_mode=False)
    assert "async_mode=False" in msg


@pytest.mark.asyncio
async def test_scorer_progress_meter(mock_scorer, capsys):
    """Test progress meter display"""
    # Test with display_meter=True
    with scorer_progress_meter(mock_scorer, display_meter=True):
        pass

    # Test with display_meter=False
    with scorer_progress_meter(mock_scorer, display_meter=False):
        pass


def test_parse_response_json_valid():
    """Test parsing valid JSON responses"""
    valid_json = '{"score": 0.8, "reason": "test"}'
    result = parse_response_json(valid_json)
    assert result == {"score": 0.8, "reason": "test"}

    # Test JSON with surrounding text
    text_with_json = 'Some text {"score": 0.9} more text'
    result = parse_response_json(text_with_json)
    assert result == {"score": 0.9}


def test_parse_response_json_invalid(mock_scorer):
    """
    Test parsing invalid JSON responses, but still completes the JSON parsing without error.
    """
    invalid_json = '{"score": 0.8, "reason": "test"'  # Missing closing brace

    # the parse_response_json function should add the missing brace and parse the JSON
    assert parse_response_json(invalid_json, scorer=mock_scorer) == {
        "score": 0.8,
        "reason": "test",
    }
    assert mock_scorer.error is None


def test_parse_response_json_missing_beginning_brace(mock_scorer):
    """
    Test that parse_response_json raises an error when JSON is missing opening brace.
    """
    invalid_json = 'score": 0.8, "reason": "test}'  # Missing opening brace

    with pytest.raises(ValueError) as exc_info:
        parse_response_json(invalid_json, scorer=mock_scorer)

    assert "Evaluation LLM outputted an invalid JSON" in str(exc_info.value)
    assert mock_scorer.error is not None


def test_create_verbose_logs(mock_scorer, capsys):
    """Test verbose logs creation"""
    steps = ["Step 1", "Step 2", "Final step"]
    logs = create_verbose_logs(mock_scorer, steps)

    assert "Step 1" in logs
    assert "Step 2" in logs

    # Check printed output when verbose_mode is True
    captured = capsys.readouterr()
    assert "MockScorer Verbose Logs" in captured.out

    # Test with verbose_mode=False
    mock_scorer.verbose_mode = False
    create_verbose_logs(mock_scorer, steps)
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.asyncio
async def test_get_or_create_event_loop():
    """Test event loop creation and retrieval"""
    # Remove the is_running check since the loop will be running under pytest-asyncio
    loop = get_or_create_event_loop()
    assert isinstance(loop, asyncio.AbstractEventLoop)

    # Test with running loop
    async def dummy_task():
        pass

    loop.create_task(dummy_task())
    loop2 = get_or_create_event_loop()
    assert loop2 is not None

    assert loop.is_running()


def test_print_verbose_logs(capsys):
    """Test verbose logs printing"""
    metric = "TestMetric"
    logs = "Test logs content"
    print_verbose_logs(metric, logs)

    captured = capsys.readouterr()
    assert "TestMetric Verbose Logs" in captured.out
    assert "Test logs content" in captured.out
