import pytest
from judgeval.judges.together_judge import TogetherJudge
from pydantic import BaseModel


def test_together_judge_invalid_input():
    """Test that TogetherJudge raises TypeError for invalid input"""
    judge = TogetherJudge()
    invalid_input = 123  # Neither string nor list

    with pytest.raises(TypeError) as exc_info:
        judge.generate(invalid_input)

    assert "Input must be a string or a list of dictionaries" in str(exc_info.value)


def test_together_judge_valid_string_input(mocker):
    """Test that TogetherJudge handles string input correctly"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.together_judge.fetch_together_api_response",
        return_value="mocked response",
    )
    judge = TogetherJudge()
    test_input = "test prompt"

    # Execute
    result = judge.generate(test_input)

    # Verify
    assert isinstance(result, str)
    assert result == "mocked response"
    mock_fetch.assert_called_once()

    # Verify the conversation format
    call_args = mock_fetch.call_args[
        0
    ]  # Using args instead of kwargs based on your implementation
    assert call_args[0] == judge.model
    assert len(call_args[1]) == 2  # Base conversation + user input
    assert call_args[1][-1] == {"role": "user", "content": test_input}


def test_together_judge_valid_list_input(mocker):
    """Test that TogetherJudge handles list input correctly"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.together_judge.fetch_together_api_response",
        return_value="mocked response",
    )
    judge = TogetherJudge()
    test_input = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]

    # Execute
    result = judge.generate(test_input)

    # Verify
    assert isinstance(result, str)
    assert result == "mocked response"
    mock_fetch.assert_called_once()

    # Verify the conversation format is passed through directly
    call_args = mock_fetch.call_args[0]
    assert call_args[0] == judge.model
    assert call_args[1] == test_input


def test_together_judge_with_schema(mocker):
    """Test that TogetherJudge correctly handles schema parameter"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.together_judge.fetch_together_api_response",
        return_value="mocked response",
    )
    judge = TogetherJudge()
    test_input = "test prompt"

    class TestSchema(BaseModel):
        response: str
        confidence: float

    # Execute
    result = judge.generate(test_input, schema=TestSchema)

    # Verify
    assert isinstance(result, str)
    assert result == "mocked response"
    mock_fetch.assert_called_once()

    # Verify schema was passed through
    call_args = mock_fetch.call_args[1]
    assert call_args["response_format"] == TestSchema


def test_together_judge_initialization():
    """Test that TogetherJudge initializes with correct parameters"""
    model_name = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    judge = TogetherJudge(model=model_name)

    assert judge.model == model_name
    assert isinstance(judge.kwargs, dict)


@pytest.mark.asyncio
async def test_together_judge_invalid_input_async():
    """Test that TogetherJudge raises TypeError for invalid input in async mode"""
    judge = TogetherJudge()
    invalid_input = 123  # Neither string nor list

    with pytest.raises(TypeError) as exc_info:
        await judge.a_generate(invalid_input)

    assert "Input must be a string or a list of dictionaries" in str(exc_info.value)


@pytest.mark.asyncio
async def test_together_judge_valid_string_input_async(mocker):
    """Test that TogetherJudge handles string input correctly in async mode"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.together_judge.afetch_together_api_response",
        return_value="mocked response",
    )
    judge = TogetherJudge()
    test_input = "test prompt"

    # Execute
    result = await judge.a_generate(test_input)

    # Verify
    assert isinstance(result, str)
    assert result == "mocked response"
    mock_fetch.assert_called_once()

    # Verify the conversation format
    call_args = mock_fetch.call_args[
        0
    ]  # Using args instead of kwargs based on your implementation
    assert call_args[0] == judge.model
    assert len(call_args[1]) == 2  # Base conversation + user input
    assert call_args[1][-1] == {"role": "user", "content": test_input}


@pytest.mark.asyncio
async def test_together_judge_valid_list_input_async(mocker):
    """Test that TogetherJudge handles list input correctly in async mode"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.together_judge.afetch_together_api_response",
        return_value="mocked response",
    )
    judge = TogetherJudge()
    test_input = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]

    # Execute
    result = await judge.a_generate(test_input)

    # Verify
    assert isinstance(result, str)
    assert result == "mocked response"
    mock_fetch.assert_called_once()

    # Verify the conversation format is passed through directly
    call_args = mock_fetch.call_args[0]
    assert call_args[0] == judge.model
    assert call_args[1] == test_input


@pytest.mark.asyncio
async def test_together_judge_with_schema_async(mocker):
    """Test that TogetherJudge correctly handles schema parameter in async mode"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.together_judge.afetch_together_api_response",
        return_value="mocked response",
    )
    judge = TogetherJudge()
    test_input = "test prompt"

    class TestSchema(BaseModel):
        response: str
        confidence: float

    # Execute
    result = await judge.a_generate(test_input, schema=TestSchema)

    # Verify
    assert isinstance(result, str)
    assert result == "mocked response"
    mock_fetch.assert_called_once()

    # Verify schema was passed through
    call_args = mock_fetch.call_args[1]
    assert call_args["response_format"] == TestSchema


def test_together_judge_load_model():
    """Test that TogetherJudge.load_model returns the correct model"""
    model_name = "test-model"
    judge = TogetherJudge(model=model_name)
    assert judge.load_model() == model_name


def test_together_judge_get_model_name():
    """Test that TogetherJudge.get_model_name returns the correct model name"""
    model_name = "test-model"
    judge = TogetherJudge(model=model_name)
    assert judge.get_model_name() == model_name
