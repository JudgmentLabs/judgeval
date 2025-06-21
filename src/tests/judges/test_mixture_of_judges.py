import pytest
from judgeval.judges.mixture_of_judges import (
    build_dynamic_mixture_prompt,
    MixtureOfJudges,
)
from pydantic import BaseModel


# Test schemas and fixtures at the top
class SampleResponseSchema(BaseModel):
    answer: str
    confidence: float


class SampleAggregationSchema(BaseModel):
    final_answer: str
    agreement_level: float


@pytest.fixture
def mixture_judge():
    return MixtureOfJudges(models=["model1", "model2"], aggregator="gpt-4")


# Group 1: build_dynamic_mixture_prompt tests
def test_build_dynamic_mixture_prompt_success():
    sample_responses = ["Response 1", "Response 2"]
    valid_custom_prompt = "You are a helpful judge synthesizing responses."
    valid_conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    # Test with only judge responses
    result1 = build_dynamic_mixture_prompt(sample_responses)
    assert isinstance(result1, list)
    assert len(result1) == 6  # Default conversation has 6 messages
    assert result1[-1]["content"].endswith(
        "Response 1\n# Judge 2's response: #\nResponse 2\n## End of Judge Responses ##\nSynthesized response:\n"
    )

    # Test with custom system prompt
    result2 = build_dynamic_mixture_prompt(
        sample_responses, custom_system_prompt=valid_custom_prompt
    )
    assert isinstance(result2, list)
    assert len(result2) == 6  # Same length as default
    assert result2[0]["content"].startswith(valid_custom_prompt)
    assert (
        "**IMPORTANT**: IF THE JUDGE RESPONSES ARE IN JSON FORMAT"
        in result2[0]["content"]
    )

    # Test with custom conversation history
    result3 = build_dynamic_mixture_prompt(
        sample_responses, custom_conversation_history=valid_conversation
    )
    assert isinstance(result3, list)
    assert len(result3) == 4  # 3 original messages + 1 new message
    assert result3[0:3] == valid_conversation
    assert "## Start of Judge Responses ##" in result3[-1]["content"]

    # Test with both custom prompt and conversation history
    # This tests that custom_conversation_history overrides custom_system_prompt
    result4 = build_dynamic_mixture_prompt(
        sample_responses,
        custom_system_prompt=valid_custom_prompt,
        custom_conversation_history=valid_conversation,
    )
    assert isinstance(result4, list)
    assert len(result4) == 4  # 3 original messages + 1 new message
    assert result4[0:3] == valid_conversation
    assert (
        valid_conversation[0]["content"] in result4[0]["content"]
    )  # Verify custom prompt is included
    assert "## Start of Judge Responses ##" in result4[-1]["content"]


def test_build_dynamic_mixture_prompt_validation():
    sample_responses = ["Response 1", "Response 2"]

    # Test invalid system prompt type
    with pytest.raises(TypeError, match="Custom system prompt must be a string"):
        build_dynamic_mixture_prompt(sample_responses, custom_system_prompt=123)

    # Test empty system prompt
    with pytest.raises(ValueError, match="Custom system prompt cannot be empty"):
        build_dynamic_mixture_prompt(sample_responses, custom_system_prompt="")

    # Test invalid conversation history format
    invalid_conversation = ["not a dict"]
    with pytest.raises(
        TypeError, match="Custom conversation history must be a list of dictionaries"
    ):
        build_dynamic_mixture_prompt(
            sample_responses, custom_conversation_history=invalid_conversation
        )

    # Test missing required keys in conversation messages
    invalid_messages = [{"role": "user"}]  # missing 'content'
    with pytest.raises(
        ValueError, match="Each message must have 'role' and 'content' keys"
    ):
        build_dynamic_mixture_prompt(
            sample_responses, custom_conversation_history=invalid_messages
        )

    # Test invalid types for role and content
    invalid_types = [{"role": 123, "content": "test"}]
    with pytest.raises(TypeError, match="Message role and content must be strings"):
        build_dynamic_mixture_prompt(
            sample_responses, custom_conversation_history=invalid_types
        )

    # Test invalid role value
    invalid_role = [{"role": "invalid_role", "content": "test"}]
    with pytest.raises(
        ValueError, match="Message role must be one of: 'system', 'user', 'assistant'"
    ):
        build_dynamic_mixture_prompt(
            sample_responses, custom_conversation_history=invalid_role
        )


# Group 2: Core functionality tests
def test_valid_string_input(mocker, mixture_judge):
    # Set up mocks
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_chat_completion"
    )
    mock_build_prompt = mocker.patch(
        "judgeval.judges.mixture_of_judges.build_dynamic_mixture_prompt"
    )

    # Sample data
    input_str = "test query"
    expected_convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_str},
    ]
    mock_responses = ["response1", "response2"]
    mock_mixture_prompt = [
        {"role": "system", "content": "You are an aggregator"},
        {"role": "user", "content": "Synthesize these responses..."},
    ]

    # Set up mock returns
    mock_get_completion.return_value = mock_responses
    mock_build_prompt.return_value = mock_mixture_prompt
    mock_chat_completion.return_value = "aggregated response"

    # Call the method
    mixture_judge.generate(input=input_str)

    # Verify interactions
    mock_get_completion.assert_called_once_with(
        models=mixture_judge.models,
        messages=[expected_convo] * len(mixture_judge.models),
        response_formats=[None] * len(mixture_judge.models),
    )

    mock_build_prompt.assert_called_once_with(
        mock_responses,
        mixture_judge.kwargs.get("custom_prompt"),
        mixture_judge.kwargs.get("custom_conversation"),
    )

    mock_chat_completion.assert_called_once_with(
        model_type=mixture_judge.aggregator,
        messages=mock_mixture_prompt,
        response_format=None,
    )


def test_valid_conversation_input(mocker, mixture_judge):
    # Set up mocks
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_chat_completion"
    )
    mock_build_prompt = mocker.patch(
        "judgeval.judges.mixture_of_judges.build_dynamic_mixture_prompt"
    )

    # Sample data
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    mock_responses = ["response1", "response2"]
    mock_mixture_prompt = [
        {"role": "system", "content": "You are an aggregator"},
        {"role": "user", "content": "Synthesize these responses..."},
    ]

    # Set up mock returns
    mock_get_completion.return_value = mock_responses
    mock_build_prompt.return_value = mock_mixture_prompt
    mock_chat_completion.return_value = "aggregated response"

    # Call the method
    mixture_judge.generate(input=conversation)

    # Verify interactions
    mock_get_completion.assert_called_once_with(
        models=mixture_judge.models,
        messages=[conversation] * len(mixture_judge.models),
        response_formats=[None] * len(mixture_judge.models),
    )

    mock_build_prompt.assert_called_once_with(
        mock_responses,
        mixture_judge.kwargs.get("custom_prompt"),
        mixture_judge.kwargs.get("custom_conversation"),
    )

    mock_chat_completion.assert_called_once_with(
        model_type=mixture_judge.aggregator,
        messages=mock_mixture_prompt,
        response_format=None,
    )


def test_return_value(mocker, mixture_judge):
    # Set up mocks
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_chat_completion"
    )
    mock_build_prompt = mocker.patch(
        "judgeval.judges.mixture_of_judges.build_dynamic_mixture_prompt"
    )

    # Set up return values
    mock_get_completion.return_value = ["response1", "response2"]
    mock_build_prompt.return_value = [{"role": "user", "content": "test"}]
    expected_response = "aggregated response"
    mock_chat_completion.return_value = expected_response

    # Call the method
    result = mixture_judge.generate(input="test query")

    # Verify the result is exactly what get_chat_completion returned
    assert result == expected_response


def test_schema_handling(mocker, mixture_judge):
    # Set up mocks
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_chat_completion"
    )
    mock_build_prompt = mocker.patch(
        "judgeval.judges.mixture_of_judges.build_dynamic_mixture_prompt"
    )

    # Set up schemas
    response_schema = SampleResponseSchema
    aggregation_schema = SampleAggregationSchema

    # Set up return values
    mock_get_completion.return_value = ["response1", "response2"]
    mock_build_prompt.return_value = [{"role": "user", "content": "test"}]
    mock_chat_completion.return_value = "aggregated response"

    # Call the method with schemas
    mixture_judge.generate(
        input="test query",
        response_schema=response_schema,
        aggregation_schema=aggregation_schema,
    )

    # Verify schemas were passed correctly to both completion functions
    mock_get_completion.assert_called_once()
    assert mock_get_completion.call_args[1]["response_formats"] == [
        response_schema
    ] * len(mixture_judge.models)

    mock_chat_completion.assert_called_once()
    assert mock_chat_completion.call_args[1]["response_format"] == aggregation_schema


# Group 3: Error handling tests
def test_invalid_input_type(mixture_judge):
    with pytest.raises(
        TypeError, match="Input must be a string or a list of dictionaries"
    ):
        mixture_judge.generate(input=123)


def test_invalid_input_type_none(mixture_judge):
    with pytest.raises(
        TypeError, match="Input must be a string or a list of dictionaries"
    ):
        mixture_judge.generate(input=None)


def test_get_completion_error(mocker, mixture_judge):
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_completion_multiple_models"
    )
    mock_get_completion.side_effect = Exception("API error")

    with pytest.raises(Exception, match="API error"):
        mixture_judge.generate(input="test query")


def test_get_chat_completion_error(mocker, mixture_judge):
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.get_chat_completion"
    )

    # Mock successful initial completions
    mock_get_completion.return_value = ["response1", "response2"]

    # Mock aggregator error
    mock_chat_completion.side_effect = Exception("Aggregator error")

    with pytest.raises(Exception, match="Aggregator error"):
        mixture_judge.generate(input="test query")


# Group 2: Core functionality tests for a_generate
@pytest.mark.asyncio
async def test_valid_string_input_async(mocker, mixture_judge):
    # Set up mocks
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_chat_completion"
    )
    mock_build_prompt = mocker.patch(
        "judgeval.judges.mixture_of_judges.build_dynamic_mixture_prompt"
    )

    # Sample data
    input_str = "test query"
    expected_convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_str},
    ]
    mock_responses = ["response1", "response2"]
    mock_mixture_prompt = [
        {"role": "system", "content": "You are an aggregator"},
        {"role": "user", "content": "Synthesize these responses..."},
    ]

    # Set up mock returns
    mock_get_completion.return_value = mock_responses
    mock_build_prompt.return_value = mock_mixture_prompt
    mock_chat_completion.return_value = "aggregated response"

    # Call the method
    await mixture_judge.a_generate(input=input_str)

    # Verify interactions
    mock_get_completion.assert_called_once_with(
        models=mixture_judge.models,
        messages=[expected_convo] * len(mixture_judge.models),
        response_formats=[None] * len(mixture_judge.models),
    )

    mock_build_prompt.assert_called_once_with(
        mock_responses,
        mixture_judge.kwargs.get("custom_prompt"),
        mixture_judge.kwargs.get("custom_conversation"),
    )

    mock_chat_completion.assert_called_once_with(
        model_type=mixture_judge.aggregator,
        messages=mock_mixture_prompt,
        response_format=None,
    )


@pytest.mark.asyncio
async def test_valid_conversation_input_async(mocker, mixture_judge):
    # Set up mocks
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_chat_completion"
    )
    mock_build_prompt = mocker.patch(
        "judgeval.judges.mixture_of_judges.build_dynamic_mixture_prompt"
    )

    # Sample data
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    mock_responses = ["response1", "response2"]
    mock_mixture_prompt = [
        {"role": "system", "content": "You are an aggregator"},
        {"role": "user", "content": "Synthesize these responses..."},
    ]

    # Set up mock returns
    mock_get_completion.return_value = mock_responses
    mock_build_prompt.return_value = mock_mixture_prompt
    mock_chat_completion.return_value = "aggregated response"

    # Call the method
    await mixture_judge.a_generate(input=conversation)

    # Verify interactions
    mock_get_completion.assert_called_once_with(
        models=mixture_judge.models,
        messages=[conversation] * len(mixture_judge.models),
        response_formats=[None] * len(mixture_judge.models),
    )

    mock_build_prompt.assert_called_once_with(
        mock_responses,
        mixture_judge.kwargs.get("custom_prompt"),
        mixture_judge.kwargs.get("custom_conversation"),
    )

    mock_chat_completion.assert_called_once_with(
        model_type=mixture_judge.aggregator,
        messages=mock_mixture_prompt,
        response_format=None,
    )


@pytest.mark.asyncio
async def test_return_value_async(mocker, mixture_judge):
    # Set up mocks
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_chat_completion"
    )
    mock_build_prompt = mocker.patch(
        "judgeval.judges.mixture_of_judges.build_dynamic_mixture_prompt"
    )

    # Set up return values
    mock_get_completion.return_value = ["response1", "response2"]
    mock_build_prompt.return_value = [{"role": "user", "content": "test"}]
    expected_response = "aggregated response"
    mock_chat_completion.return_value = expected_response

    # Call the method
    result = await mixture_judge.a_generate(input="test query")

    # Verify the result is exactly what get_chat_completion returned
    assert result == expected_response


def test_load_model(mixture_judge):
    """Test that load_model returns the list of models."""
    expected_models = ["model1", "model2"]
    assert mixture_judge.load_model() == expected_models


def test_get_model_name(mixture_judge):
    """Test that get_model_name returns the list of models."""
    expected_models = ["model1", "model2"]
    assert mixture_judge.get_model_name() == expected_models


def test_model_list_consistency(mixture_judge):
    """Test that both methods return the same list of models."""
    assert mixture_judge.load_model() == mixture_judge.get_model_name()


@pytest.mark.asyncio
async def test_invalid_input_type_async(mixture_judge):
    with pytest.raises(
        TypeError, match="Input must be a string or a list of dictionaries"
    ):
        await mixture_judge.a_generate(input=123)


@pytest.mark.asyncio
async def test_invalid_input_type_none_async(mixture_judge):
    with pytest.raises(
        TypeError, match="Input must be a string or a list of dictionaries"
    ):
        await mixture_judge.a_generate(input=None)


@pytest.mark.asyncio
async def test_get_completion_error_async(mocker, mixture_judge):
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_completion_multiple_models"
    )
    mock_get_completion.side_effect = Exception("API error")

    with pytest.raises(Exception, match="API error"):
        await mixture_judge.a_generate(input="test query")


@pytest.mark.asyncio
async def test_get_chat_completion_error_async(mocker, mixture_judge):
    mock_get_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_completion_multiple_models"
    )
    mock_chat_completion = mocker.patch(
        "judgeval.judges.mixture_of_judges.aget_chat_completion"
    )

    # Mock successful initial completions
    mock_get_completion.return_value = ["response1", "response2"]

    # Mock aggregator error
    mock_chat_completion.side_effect = Exception("Aggregator error")

    with pytest.raises(Exception, match="Aggregator error"):
        await mixture_judge.a_generate(input="test query")
