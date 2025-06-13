import pytest
from judgeval.judges.base_judge import JudgevalJudge
from judgeval.judges.litellm_judge import LiteLLMJudge
from pydantic import BaseModel


# Base Judge
def test_cannot_instantiate_abstract_class():
    """Test that the abstract class cannot be instantiated directly"""
    with pytest.raises(TypeError):
        JudgevalJudge()


def test_concrete_implementation():
    """Test that a concrete implementation works as expected"""

    # Create a concrete implementation for testing
    class ConcreteJudge(JudgevalJudge):
        def load_model(self, *args, **kwargs):
            return "mock_model"

        def generate(self, *args, **kwargs) -> str:
            return "generated_text"

        async def a_generate(self, *args, **kwargs) -> str:
            return "async_generated_text"

        def get_model_name(self, *args, **kwargs) -> str:
            return "mock_model_name"

    # Test initialization
    judge = ConcreteJudge(model_name="test_model")
    assert judge.model_name == "test_model"
    assert judge.model == "mock_model"

    # Test methods
    assert judge.generate() == "generated_text"
    assert judge.get_model_name() == "mock_model_name"


# LiteLLM Judge Tests
def test_litellm_judge_invalid_input():
    """Test that LiteLLMJudge raises TypeError for invalid input"""
    judge = LiteLLMJudge()
    invalid_input = 123  # Neither string nor list

    with pytest.raises(TypeError) as exc_info:
        judge.generate(invalid_input)

    assert "Input must be a string or a list of dictionaries" in str(exc_info.value)


def test_litellm_judge_valid_string_input(mocker):
    """Test that LiteLLMJudge handles string input correctly"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.litellm_judge.fetch_litellm_api_response",
        return_value="mocked response",
    )
    judge = LiteLLMJudge()
    test_input = "test prompt"

    # Execute
    result = judge.generate(test_input)

    # Verify
    assert isinstance(result, str)
    assert result == "mocked response"
    mock_fetch.assert_called_once()

    # Verify the conversation format
    call_args = mock_fetch.call_args[1]
    assert call_args["model"] == judge.model
    assert len(call_args["messages"]) == 2  # Base system message + user input
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1] == {"role": "user", "content": test_input}


def test_litellm_judge_valid_list_input(mocker):
    """Test that LiteLLMJudge handles list input correctly"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.litellm_judge.fetch_litellm_api_response",
        return_value="mocked response",
    )
    judge = LiteLLMJudge()
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
    call_args = mock_fetch.call_args[1]
    assert call_args["model"] == judge.model
    assert call_args["messages"] == test_input


def test_litellm_judge_with_schema(mocker):
    """Test that LiteLLMJudge correctly handles schema parameter"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.litellm_judge.fetch_litellm_api_response",
        return_value="mocked response",
    )
    judge = LiteLLMJudge()
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


def test_litellm_judge_load_model():
    """Test that LiteLLMJudge.load_model returns the correct model"""
    judge = LiteLLMJudge(model="test-model")
    assert judge.load_model() == "test-model"


def test_litellm_judge_get_model_name():
    """Test that LiteLLMJudge.get_model_name returns the correct model name"""
    judge = LiteLLMJudge(model="test-model")
    assert judge.get_model_name() == "test-model"


# Async LiteLLM Judge Tests
@pytest.mark.asyncio
async def test_litellm_judge_invalid_input_async():
    """Test that LiteLLMJudge raises TypeError for invalid input in async mode"""
    judge = LiteLLMJudge()
    invalid_input = 123  # Neither string nor list

    with pytest.raises(TypeError) as exc_info:
        await judge.a_generate(invalid_input)

    assert "Input must be a string or a list of dictionaries" in str(exc_info.value)


@pytest.mark.asyncio
async def test_litellm_judge_valid_string_input_async(mocker):
    """Test that LiteLLMJudge handles string input correctly in async mode"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.litellm_judge.afetch_litellm_api_response",
        return_value="mocked response",
    )
    judge = LiteLLMJudge()
    test_input = "test prompt"

    # Execute
    result = await judge.a_generate(test_input)

    # Verify
    assert isinstance(result, str)
    assert result == "mocked response"
    mock_fetch.assert_called_once()

    # Verify the conversation format
    call_args = mock_fetch.call_args[1]
    assert call_args["model"] == judge.model
    assert len(call_args["messages"]) == 2  # Base system message + user input
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1] == {"role": "user", "content": test_input}


@pytest.mark.asyncio
async def test_litellm_judge_valid_list_input_async(mocker):
    """Test that LiteLLMJudge handles list input correctly in async mode"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.litellm_judge.afetch_litellm_api_response",
        return_value="mocked response",
    )
    judge = LiteLLMJudge()
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
    call_args = mock_fetch.call_args[1]
    assert call_args["model"] == judge.model
    assert call_args["messages"] == test_input


@pytest.mark.asyncio
async def test_litellm_judge_with_schema_async(mocker):
    """Test that LiteLLMJudge correctly handles schema parameter in async mode"""
    # Setup
    mock_fetch = mocker.patch(
        "judgeval.judges.litellm_judge.afetch_litellm_api_response",
        return_value="mocked response",
    )
    judge = LiteLLMJudge()
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
