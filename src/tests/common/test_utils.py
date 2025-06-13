import pytest
from judgeval.common.utils import (
    afetch_together_api_response,
    fetch_custom_litellm_api_response,
    afetch_custom_litellm_api_response,
    CustomModelParameters,
    aget_completion_multiple_models,
    query_together_api_multiple_calls,
    aquery_together_api_multiple_calls,
    get_chat_completion,
    aget_chat_completion,
    get_completion_multiple_models,
    fetch_together_api_response,
    ChatCompletionRequest,
    fetch_litellm_api_response,
)
from judgeval.constants import TOGETHER_SUPPORTED_MODELS
import litellm

LITELLM_SUPPORTED_MODELS = set(litellm.model_list)

# Test data
TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

TEST_BATCHED_MESSAGES = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Japan?"},
    ],
]


class MockResponse:
    def __init__(self, content):
        self.choices = [
            type("Choice", (), {"message": type("Message", (), {"content": content})()})
        ]


@pytest.fixture
def mock_apis(mocker):
    # Mock Together API
    together_mock = mocker.patch("judgeval.common.utils.together_client")
    together_mock.chat.completions.create.return_value = MockResponse("Paris")

    # Mock Async Together API
    async_together_mock = mocker.patch("judgeval.common.utils.async_together_client")
    async_together_mock.chat.completions.create = mocker.AsyncMock(
        return_value=MockResponse("Paris")
    )

    # Mock LiteLLM
    litellm_mock = mocker.patch("judgeval.common.utils.litellm")
    litellm_mock.completion.return_value = MockResponse("Tokyo")
    litellm_mock.acompletion = mocker.AsyncMock(return_value=MockResponse("Tokyo"))

    return together_mock, async_together_mock, litellm_mock


# Together API Tests
def test_fetch_together_api_response_success(mock_apis):
    together_mock, _, _ = mock_apis
    model = TOGETHER_SUPPORTED_MODELS[0]

    response = fetch_together_api_response(model, TEST_MESSAGES)

    assert response == "Paris"
    together_mock.chat.completions.create.assert_called_once()


def test_fetch_together_api_response_invalid_model(mock_apis):
    with pytest.raises(
        ValueError, match="Model invalid_model is not in the list of supported models."
    ):
        fetch_together_api_response("invalid_model", TEST_MESSAGES)


@pytest.mark.asyncio
async def test_afetch_together_api_response_success(mock_apis):
    _, async_together_mock, _ = mock_apis
    model = TOGETHER_SUPPORTED_MODELS[0]

    response = await afetch_together_api_response(model, TEST_MESSAGES)

    assert response == "Paris"
    async_together_mock.chat.completions.create.assert_called_once()


# LiteLLM API Tests
def test_fetch_litellm_api_response_success(mock_apis):
    _, _, litellm_mock = mock_apis
    model = list(LITELLM_SUPPORTED_MODELS)[0]

    response = fetch_litellm_api_response(model, TEST_MESSAGES)

    assert response == "Tokyo"
    litellm_mock.completion.assert_called_once()


def test_fetch_custom_litellm_api_response(mock_apis):
    _, _, litellm_mock = mock_apis
    custom_params = CustomModelParameters(
        model_name="custom-model",
        secret_key="test-key",
        litellm_base_url="http://test.com",
    )

    response = fetch_custom_litellm_api_response(custom_params, TEST_MESSAGES)

    assert response == "Tokyo"
    litellm_mock.completion.assert_called_once_with(
        model="custom-model",
        messages=TEST_MESSAGES,
        api_key="test-key",
        base_url="http://test.com",
    )


@pytest.mark.asyncio
async def test_afetch_custom_litellm_api_response(mock_apis):
    _, _, litellm_mock = mock_apis
    custom_params = CustomModelParameters(
        model_name="custom-model",
        secret_key="test-key",
        litellm_base_url="http://test.com",
    )

    response = await afetch_custom_litellm_api_response(custom_params, TEST_MESSAGES)

    assert response == "Tokyo"
    litellm_mock.acompletion.assert_called_once()


# Multiple Calls Tests
def test_query_together_api_multiple_calls(mock_apis, mocker):
    together_mock, _, _ = mock_apis
    model = TOGETHER_SUPPORTED_MODELS[0]

    responses = query_together_api_multiple_calls(
        [model] * 2, TEST_BATCHED_MESSAGES, [None] * 2
    )

    assert len(responses) == 2
    assert all(r == "Paris" for r in responses)
    assert together_mock.chat.completions.create.call_count == 2


def test_query_together_api_multiple_calls_with_error(mock_apis):
    together_mock, _, _ = mock_apis
    together_mock.chat.completions.create.side_effect = Exception("API Error")
    model = TOGETHER_SUPPORTED_MODELS[0]

    responses = query_together_api_multiple_calls([model], [TEST_MESSAGES], [None])

    assert responses == [None]


@pytest.mark.asyncio
async def test_aquery_together_api_multiple_calls(mock_apis):
    _, async_together_mock, _ = mock_apis
    model = TOGETHER_SUPPORTED_MODELS[0]

    responses = await aquery_together_api_multiple_calls(
        [model] * 2, TEST_BATCHED_MESSAGES, [None] * 2
    )

    assert len(responses) == 2
    assert all(r == "Paris" for r in responses)
    assert async_together_mock.chat.completions.create.call_count == 2


# High-level Interface Tests
def test_get_chat_completion_together(mock_apis):
    together_mock, _, _ = mock_apis
    model = TOGETHER_SUPPORTED_MODELS[0]

    response = get_chat_completion(model, TEST_MESSAGES)

    assert response == "Paris"
    together_mock.chat.completions.create.assert_called_once()


def test_get_chat_completion_batched(mock_apis):
    together_mock, _, _ = mock_apis
    model = TOGETHER_SUPPORTED_MODELS[0]

    responses = get_chat_completion(model, TEST_BATCHED_MESSAGES, batched=True)

    assert len(responses) == 2
    assert all(r == "Paris" for r in responses)
    assert together_mock.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_aget_chat_completion(mock_apis):
    _, async_together_mock, _ = mock_apis
    model = TOGETHER_SUPPORTED_MODELS[0]

    response = await aget_chat_completion(model, TEST_MESSAGES)

    assert response == "Paris"
    async_together_mock.chat.completions.create.assert_called_once()


# Multiple Models Tests
def test_get_completion_multiple_models(mock_apis):
    together_mock, _, litellm_mock = mock_apis
    together_model = TOGETHER_SUPPORTED_MODELS[0]
    litellm_model = list(LITELLM_SUPPORTED_MODELS)[0]

    responses = get_completion_multiple_models(
        [together_model, litellm_model], [TEST_MESSAGES] * 2
    )

    assert len(responses) == 2
    assert responses[0] == "Paris"
    assert responses[1] == "Tokyo"
    together_mock.chat.completions.create.assert_called_once()
    litellm_mock.completion.assert_called_once()


@pytest.mark.asyncio
async def test_aget_completion_multiple_models(mock_apis):
    _, async_together_mock, litellm_mock = mock_apis
    together_model = TOGETHER_SUPPORTED_MODELS[0]
    litellm_model = list(LITELLM_SUPPORTED_MODELS)[0]

    responses = await aget_completion_multiple_models(
        [together_model, litellm_model], [TEST_MESSAGES] * 2
    )

    assert len(responses) == 2
    assert responses[0] == "Paris"
    assert responses[1] == "Tokyo"
    async_together_mock.chat.completions.create.assert_called_once()
    litellm_mock.acompletion.assert_called_once()


# Error Cases
def test_get_completion_multiple_models_length_mismatch(mock_apis):
    with pytest.raises(
        ValueError, match="Number of models and messages must be the same"
    ):
        get_completion_multiple_models(["model1", "model2"], [TEST_MESSAGES])


def test_get_completion_multiple_models_invalid_model(mock_apis):
    with pytest.raises(ValueError, match="not supported by Litellm or TogetherAI"):
        get_completion_multiple_models(["invalid_model"], [TEST_MESSAGES])


def test_response_format_handling(mock_apis):
    together_mock, _, _ = mock_apis
    model = TOGETHER_SUPPORTED_MODELS[0]
    response_format = dict()

    fetch_together_api_response(model, TEST_MESSAGES, response_format)

    together_mock.chat.completions.create.assert_called_once_with(
        model=model, messages=TEST_MESSAGES, response_format=response_format
    )


# Together API ValueError Tests
def test_fetch_together_api_response_value_errors():
    # Test with invalid model
    with pytest.raises(
        ValueError, match="Model invalid_model is not in the list of supported models"
    ):
        ChatCompletionRequest(model="invalid_model", messages=TEST_MESSAGES)

    # Test with empty messages
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        ChatCompletionRequest(model=TOGETHER_SUPPORTED_MODELS[0], messages=[])

    # Test with invalid message format
    with pytest.raises(ValueError, match="Message missing required 'role' field"):
        ChatCompletionRequest(
            model=TOGETHER_SUPPORTED_MODELS[0], messages=[{"invalid": "format"}]
        )

    # Test with None messages
    with pytest.raises(ValueError, match="Input should be a valid list"):
        ChatCompletionRequest(model=TOGETHER_SUPPORTED_MODELS[0], messages=None)


@pytest.mark.asyncio
async def test_afetch_together_api_response_value_errors():
    # Test with invalid model
    with pytest.raises(
        ValueError, match="Model invalid_model is not in the list of supported models"
    ):
        ChatCompletionRequest(model="invalid_model", messages=TEST_MESSAGES)

    # Test with empty messages
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        ChatCompletionRequest(model=TOGETHER_SUPPORTED_MODELS[0], messages=[])

    # Test with invalid message format
    with pytest.raises(ValueError, match="Message missing required 'role' field"):
        ChatCompletionRequest(
            model=TOGETHER_SUPPORTED_MODELS[0], messages=[{"invalid": "format"}]
        )

    # Test with None messages
    with pytest.raises(ValueError, match="Input should be a valid list"):
        ChatCompletionRequest(model=TOGETHER_SUPPORTED_MODELS[0], messages=None)


# LiteLLM API ValueError Tests
def test_fetch_litellm_api_response_value_errors():
    # Test with invalid model
    with pytest.raises(
        ValueError, match="Model invalid_model is not in the list of supported models"
    ):
        ChatCompletionRequest(model="invalid_model", messages=TEST_MESSAGES)

    # Test with empty messages
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        ChatCompletionRequest(model=list(LITELLM_SUPPORTED_MODELS)[0], messages=[])

    # Test with invalid message format
    with pytest.raises(ValueError, match="Message missing required 'role' field"):
        ChatCompletionRequest(
            model=list(LITELLM_SUPPORTED_MODELS)[0], messages=[{"invalid": "format"}]
        )

    # Test with None messages
    with pytest.raises(ValueError, match="Input should be a valid list"):
        ChatCompletionRequest(model=list(LITELLM_SUPPORTED_MODELS)[0], messages=None)


def test_fetch_custom_litellm_api_response_value_errors():
    # None parameters
    with pytest.raises(ValueError, match="Custom model parameters cannot be empty"):
        fetch_custom_litellm_api_response(None, TEST_MESSAGES)

    # Not a CustomModelParameters object
    invalid_obj = {
        "model_name": "custom-model",
        "secret_key": "test-key",
        "litellm_base_url": "http://test.com",
    }
    with pytest.raises(
        ValueError,
        match="Custom model parameters must be a CustomModelParameters object",
    ):
        fetch_custom_litellm_api_response(invalid_obj, TEST_MESSAGES)

    # Empty model name
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        CustomModelParameters(
            model_name="", secret_key="test-key", litellm_base_url="http://test.com"
        )

    # Empty secret key
    with pytest.raises(ValueError, match="Secret key cannot be empty"):
        CustomModelParameters(
            model_name="custom-model", secret_key="", litellm_base_url="http://test.com"
        )

    # Empty base URL
    with pytest.raises(ValueError, match="Litellm base URL cannot be empty"):
        CustomModelParameters(
            model_name="custom-model", secret_key="test-key", litellm_base_url=""
        )


# Multiple Calls ValueError Tests
def test_query_together_api_multiple_calls_value_errors():
    model = TOGETHER_SUPPORTED_MODELS[0]

    # Length mismatch
    with pytest.raises(
        ValueError,
        match="Number of models, messages, and response formats must be the same",
    ):
        query_together_api_multiple_calls([model], TEST_BATCHED_MESSAGES, [None, None])

    # Empty models list
    with pytest.raises(ValueError, match="Models list cannot be empty"):
        query_together_api_multiple_calls([], [], [])

    # Invalid model in list
    with pytest.raises(
        ValueError, match="Model .* is not in the list of supported models"
    ):
        query_together_api_multiple_calls(["invalid_model"], [TEST_MESSAGES], [None])


@pytest.mark.asyncio
async def test_aquery_together_api_multiple_calls_value_errors():
    model = TOGETHER_SUPPORTED_MODELS[0]

    # Length mismatch
    with pytest.raises(
        ValueError,
        match="Number of models, messages, and response formats must be the same",
    ):
        await aquery_together_api_multiple_calls(
            [model], TEST_BATCHED_MESSAGES, [None, None]
        )

    # Empty models list
    with pytest.raises(ValueError, match="Models list cannot be empty"):
        await aquery_together_api_multiple_calls([], [], [])

    # Invalid model in list
    with pytest.raises(
        ValueError, match="Model .* is not in the list of supported models"
    ):
        await aquery_together_api_multiple_calls(
            ["invalid_model"], [TEST_MESSAGES], [None]
        )


# High-level Interface ValueError Tests
def test_get_chat_completion_value_errors():
    # Invalid model
    with pytest.raises(ValueError, match="Model .* not supported"):
        get_chat_completion("invalid_model", TEST_MESSAGES)

    # Empty messages
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        get_chat_completion(TOGETHER_SUPPORTED_MODELS[0], [])

    # None messages
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        get_chat_completion(TOGETHER_SUPPORTED_MODELS[0], None)

    # Invalid batched messages format
    with pytest.raises(ValueError, match="Message missing required 'role' field"):
        get_chat_completion(
            TOGETHER_SUPPORTED_MODELS[0], [[{"invalid": "format"}]], batched=True
        )


@pytest.mark.asyncio
async def test_aget_chat_completion_value_errors():
    # Invalid model
    with pytest.raises(ValueError, match="Model .* not supported"):
        await aget_chat_completion("invalid_model", TEST_MESSAGES)

    # Empty messages
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        await aget_chat_completion(TOGETHER_SUPPORTED_MODELS[0], [])

    # None messages
    with pytest.raises(TypeError, match="Messages must be a list"):
        await aget_chat_completion(TOGETHER_SUPPORTED_MODELS[0], None)

    # Invalid batched messages format
    with pytest.raises(TypeError, match="Each batch item must be a list of messages"):
        await aget_chat_completion(
            TOGETHER_SUPPORTED_MODELS[0], [{"invalid": "format"}], batched=True
        )


# Multiple Models ValueError Tests
def test_get_completion_multiple_models_value_errors():
    # Length mismatch
    with pytest.raises(
        ValueError, match="Number of models and messages must be the same"
    ):
        get_completion_multiple_models(
            [TOGETHER_SUPPORTED_MODELS[0]], [TEST_MESSAGES, TEST_MESSAGES]
        )

    # Empty models list
    with pytest.raises(ValueError, match="Models list cannot be empty"):
        get_completion_multiple_models([], [])

    # Invalid model in list
    with pytest.raises(ValueError, match="Model .* not supported"):
        get_completion_multiple_models(["invalid_model"], [TEST_MESSAGES])

    # None messages
    with pytest.raises(TypeError, match="Each batch item must be a list of messages"):
        get_completion_multiple_models([TOGETHER_SUPPORTED_MODELS[0]], [None])


@pytest.mark.asyncio
async def test_aget_completion_multiple_models_value_errors():
    # Length mismatch
    with pytest.raises(
        ValueError, match="Number of models and messages must be the same"
    ):
        await aget_completion_multiple_models(
            [TOGETHER_SUPPORTED_MODELS[0]], [TEST_MESSAGES, TEST_MESSAGES]
        )

    # Empty models list
    with pytest.raises(ValueError, match="Models list cannot be empty"):
        await aget_completion_multiple_models([], [])

    # Invalid model in list
    with pytest.raises(ValueError, match="Model .* not supported"):
        await aget_completion_multiple_models(["invalid_model"], [TEST_MESSAGES])

    # None messages
    with pytest.raises(TypeError, match="Each batch item must be a list of messages"):
        await aget_completion_multiple_models([TOGETHER_SUPPORTED_MODELS[0]], [None])


# Response Format ValueError Tests
def test_response_format_value_errors():
    model = TOGETHER_SUPPORTED_MODELS[0]

    # Invalid response format type
    with pytest.raises(
        TypeError, match="Response format must be a dictionary or pydantic model"
    ):
        fetch_together_api_response(model, TEST_MESSAGES, response_format="invalid")
