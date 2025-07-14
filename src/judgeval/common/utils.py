"""
This file contains utility functions used in repo scripts

For API calling, we support:
    - parallelized model calls on the same prompt
    - batched model calls on different prompts

NOTE: any function beginning with 'a', e.g. 'afetch_together_api_response', is an asynchronous function
"""

# Standard library imports
import asyncio
import concurrent.futures
import os
from types import TracebackType
from judgeval.utils.requests import requests
import pprint
from typing import Any, Dict, List, Mapping, Optional, TypeAlias, Union, TypeGuard

# Third-party imports
import litellm
import pydantic
from dotenv import load_dotenv

# Local application/library-specific imports
from judgeval.clients import async_together_client, together_client
from judgeval.constants import (
    ACCEPTABLE_MODELS,
    MAX_WORKER_THREADS,
    ROOT_API,
    TOGETHER_SUPPORTED_MODELS,
    LITELLM_SUPPORTED_MODELS,
)
from judgeval.common.logger import judgeval_logger


class CustomModelParameters(pydantic.BaseModel):
    model_name: str
    secret_key: str
    litellm_base_url: str

    @pydantic.field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if not v:
            raise ValueError("Model name cannot be empty")
        return v

    @pydantic.field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if not v:
            raise ValueError("Secret key cannot be empty")
        return v

    @pydantic.field_validator("litellm_base_url")
    @classmethod
    def validate_litellm_base_url(cls, v):
        if not v:
            raise ValueError("Litellm base URL cannot be empty")
        return v


class ChatCompletionRequest(pydantic.BaseModel):
    model: str
    messages: List[Dict[str, str]]
    response_format: Optional[Union[pydantic.BaseModel, Dict[str, Any]]] = None

    @pydantic.field_validator("messages")
    @classmethod
    def validate_messages(cls, messages):
        if not messages:
            raise ValueError("Messages cannot be empty")

        for msg in messages:
            if not isinstance(msg, dict):
                raise TypeError("Message must be a dictionary")
            if "role" not in msg:
                raise ValueError("Message missing required 'role' field")
            if "content" not in msg:
                raise ValueError("Message missing required 'content' field")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    f"Invalid role '{msg['role']}'. Must be 'system', 'user', or 'assistant'"
                )

        return messages

    @pydantic.field_validator("model")
    @classmethod
    def validate_model(cls, model):
        if not model:
            raise ValueError("Model cannot be empty")
        if model not in ACCEPTABLE_MODELS:
            raise ValueError(f"Model {model} is not in the list of supported models.")
        return model

    @pydantic.field_validator("response_format", mode="before")
    @classmethod
    def validate_response_format(cls, response_format):
        if response_format is not None:
            if not isinstance(response_format, (dict, pydantic.BaseModel)):
                raise TypeError(
                    "Response format must be a dictionary or pydantic model"
                )
            # Optional: Add additional validation for required fields if needed
            # For example, checking for 'type': 'json' in OpenAI's format
        return response_format


os.environ["LITELLM_LOG"] = "DEBUG"

load_dotenv()


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def validate_api_key(judgment_api_key: str):
    """
    Validates that the user api key is valid
    """
    response = requests.post(
        f"{ROOT_API}/auth/validate_api_key/",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {judgment_api_key}",
        },
        json={},  # Empty body now
        verify=True,
    )
    if response.status_code == 200:
        return True, response.json()
    else:
        return False, response.json().get("detail", "Error validating API key")


def fetch_together_api_response(
    model: str, messages: List[Mapping], response_format: pydantic.BaseModel = None
) -> str:
    """
    Fetches a single response from the Together API for a given model and messages.
    """
    # Validate request
    if messages is None or messages == []:
        raise ValueError("Messages cannot be empty")

    request = ChatCompletionRequest(
        model=model, messages=messages, response_format=response_format
    )

    if request.response_format is not None:
        response = together_client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            response_format=request.response_format,
        )
    else:
        response = together_client.chat.completions.create(
            model=request.model,
            messages=request.messages,
        )

    return response.choices[0].message.content


async def afetch_together_api_response(
    model: str, messages: List[Mapping], response_format: pydantic.BaseModel = None
) -> str:
    """
    ASYNCHRONOUSLY Fetches a single response from the Together API for a given model and messages.
    """
    request = ChatCompletionRequest(
        model=model, messages=messages, response_format=response_format
    )

    if request.response_format is not None:
        response = await async_together_client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            response_format=request.response_format,
        )
    else:
        response = await async_together_client.chat.completions.create(
            model=request.model,
            messages=request.messages,
        )
    return response.choices[0].message.content


def query_together_api_multiple_calls(
    models: List[str],
    messages: List[List[Mapping]],
    response_formats: List[pydantic.BaseModel] | None = None,
) -> List[Union[str, None]]:
    """
    Queries the Together API for multiple calls in parallel

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query. Each inner object corresponds to a single prompt.
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: TogetherAI responses for each model and message pair in order. Any exceptions in the thread call result in a None.
    """
    # Check for empty models list
    if not models:
        raise ValueError("Models list cannot be empty")

    # Validate all models are supported
    for model in models:
        if model not in ACCEPTABLE_MODELS:
            raise ValueError(
                f"Model {model} is not in the list of supported models: {ACCEPTABLE_MODELS}."
            )

    # Validate input lengths match
    if response_formats is None:
        response_formats = [None] * len(models)
    if not (len(models) == len(messages) == len(response_formats)):
        raise ValueError(
            "Number of models, messages, and response formats must be the same"
        )

    # Validate message format
    validate_batched_chat_messages(messages)

    num_workers = int(os.getenv("NUM_WORKER_THREADS", MAX_WORKER_THREADS))
    # Initialize results to maintain ordered outputs
    out: List[str | None] = [None] * len(messages)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all queries to together API with index, gets back the response content
        futures = {
            executor.submit(
                fetch_together_api_response, model, message, response_format
            ): idx
            for idx, (model, message, response_format) in enumerate(
                zip(models, messages, response_formats)
            )
        }

        # Collect results as they complete -- result is response content
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                out[idx] = future.result()
            except Exception as e:
                judgeval_logger.error(f"Error in parallel call {idx}: {str(e)}")
                out[idx] = None
    return out


async def aquery_together_api_multiple_calls(
    models: List[str],
    messages: List[List[Mapping]],
    response_formats: List[pydantic.BaseModel] | None = None,
) -> List[Union[str, None]]:
    """
    Queries the Together API for multiple calls in parallel

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query. Each inner object corresponds to a single prompt.
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: TogetherAI responses for each model and message pair in order. Any exceptions in the thread call result in a None.
    """
    # Check for empty models list
    if not models:
        raise ValueError("Models list cannot be empty")

    # Validate all models are supported
    for model in models:
        if model not in ACCEPTABLE_MODELS:
            raise ValueError(
                f"Model {model} is not in the list of supported models: {ACCEPTABLE_MODELS}."
            )

    # Validate input lengths match
    if response_formats is None:
        response_formats = [None] * len(models)
    if not (len(models) == len(messages) == len(response_formats)):
        raise ValueError(
            "Number of models, messages, and response formats must be the same"
        )

    # Validate message format
    validate_batched_chat_messages(messages)

    out: List[Union[str, None]] = [None] * len(messages)

    async def fetch_and_store(idx, model, message, response_format):
        try:
            out[idx] = await afetch_together_api_response(
                model, message, response_format
            )
        except Exception as e:
            judgeval_logger.error(f"Error in parallel call {idx}: {str(e)}")
            out[idx] = None

    tasks = [
        fetch_and_store(idx, model, message, response_format)
        for idx, (model, message, response_format) in enumerate(
            zip(models, messages, response_formats)
        )
    ]

    await asyncio.gather(*tasks)
    return out


def fetch_litellm_api_response(
    model: str, messages: List[Mapping], response_format: pydantic.BaseModel = None
) -> str:
    """
    Fetches a single response from the Litellm API for a given model and messages.
    """
    request = ChatCompletionRequest(
        model=model, messages=messages, response_format=response_format
    )

    if request.response_format is not None:
        response = litellm.completion(
            model=request.model,
            messages=request.messages,
            response_format=request.response_format,
        )
    else:
        response = litellm.completion(
            model=request.model,
            messages=request.messages,
        )
    return response.choices[0].message.content


def fetch_custom_litellm_api_response(
    custom_model_parameters: CustomModelParameters,
    messages: List[Mapping],
    response_format: pydantic.BaseModel = None,
) -> str:
    if messages is None or messages == []:
        raise ValueError("Messages cannot be empty")

    if custom_model_parameters is None:
        raise ValueError("Custom model parameters cannot be empty")

    if not isinstance(custom_model_parameters, CustomModelParameters):
        raise ValueError(
            "Custom model parameters must be a CustomModelParameters object"
        )

    if response_format is not None:
        response = litellm.completion(
            model=custom_model_parameters.model_name,
            messages=messages,
            api_key=custom_model_parameters.secret_key,
            base_url=custom_model_parameters.litellm_base_url,
            response_format=response_format,
        )
    else:
        response = litellm.completion(
            model=custom_model_parameters.model_name,
            messages=messages,
            api_key=custom_model_parameters.secret_key,
            base_url=custom_model_parameters.litellm_base_url,
        )
    return response.choices[0].message.content


async def afetch_litellm_api_response(
    model: str, messages: List[Mapping], response_format: pydantic.BaseModel = None
) -> str:
    """
    ASYNCHRONOUSLY Fetches a single response from the Litellm API for a given model and messages.
    """
    if messages is None or messages == []:
        raise ValueError("Messages cannot be empty")

    # Add validation
    validate_chat_messages(messages)

    if model not in ACCEPTABLE_MODELS:
        raise ValueError(
            f"Model {model} is not in the list of supported models: {ACCEPTABLE_MODELS}."
        )

    if response_format is not None:
        response = await litellm.acompletion(
            model=model, messages=messages, response_format=response_format
        )
    else:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
        )
    return response.choices[0].message.content


async def afetch_custom_litellm_api_response(
    custom_model_parameters: CustomModelParameters,
    messages: List[Mapping],
    response_format: pydantic.BaseModel = None,
) -> str:
    """
    ASYNCHRONOUSLY Fetches a single response from the Litellm API for a given model and messages.
    """
    if messages is None or messages == []:
        raise ValueError("Messages cannot be empty")

    if custom_model_parameters is None:
        raise ValueError("Custom model parameters cannot be empty")

    if not isinstance(custom_model_parameters, CustomModelParameters):
        raise ValueError(
            "Custom model parameters must be a CustomModelParameters object"
        )

    if response_format is not None:
        response = await litellm.acompletion(
            model=custom_model_parameters.model_name,
            messages=messages,
            api_key=custom_model_parameters.secret_key,
            base_url=custom_model_parameters.litellm_base_url,
            response_format=response_format,
        )
    else:
        response = await litellm.acompletion(
            model=custom_model_parameters.model_name,
            messages=messages,
            api_key=custom_model_parameters.secret_key,
            base_url=custom_model_parameters.litellm_base_url,
        )
    return response.choices[0].message.content


def query_litellm_api_multiple_calls(
    models: List[str],
    messages: List[List[Mapping]],
    response_formats: List[pydantic.BaseModel] | None = None,
) -> List[Union[str, None]]:
    """
    Queries the Litellm API for multiple calls in parallel

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: Litellm responses for each model and message pair in order. Any exceptions in the thread call result in a None.
    """
    num_workers = int(os.getenv("NUM_WORKER_THREADS", MAX_WORKER_THREADS))
    # Initialize results to maintain ordered outputs
    out: List[Union[str, None]] = [None] * len(messages)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all queries to Litellm API with index, gets back the response content
        futures = {
            executor.submit(
                fetch_litellm_api_response, model, message, response_format
            ): idx
            for idx, (model, message, response_format) in enumerate(
                zip(models, messages, response_formats or [None] * len(messages))
            )
        }

        # Collect results as they complete -- result is response content
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                out[idx] = future.result()
            except Exception as e:
                judgeval_logger.error(f"Error in parallel call {idx}: {str(e)}")
                out[idx] = None
    return out


async def aquery_litellm_api_multiple_calls(
    models: List[str],
    messages: List[List[Mapping]],
    response_formats: List[pydantic.BaseModel] | None = None,
) -> List[Union[str, None]]:
    """
    Queries the Litellm API for multiple calls in parallel

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: Litellm responses for each model and message pair in order. Any exceptions in the thread call result in a None.
    """
    # Initialize results to maintain ordered outputs
    out: List[Union[str, None]] = [None] * len(messages)

    async def fetch_and_store(idx, model, message, response_format):
        try:
            out[idx] = await afetch_litellm_api_response(
                model, message, response_format
            )
        except Exception as e:
            judgeval_logger.error(f"Error in parallel call {idx}: {str(e)}")
            out[idx] = None

    tasks = [
        fetch_and_store(idx, model, message, response_format)
        for idx, (model, message, response_format) in enumerate(
            zip(models, messages, response_formats or [None] * len(messages))
        )
    ]

    await asyncio.gather(*tasks)
    return out


def validate_chat_messages(messages, batched: bool = False):
    """Validate chat message format before API call"""
    if not isinstance(messages, list):
        raise TypeError("Messages must be a list")

    for msg in messages:
        if not isinstance(msg, dict):
            if batched and not isinstance(msg, list):
                raise TypeError("Each message must be a list")
            elif not batched:
                raise TypeError("Message must be a dictionary")
        if "role" not in msg:
            raise ValueError("Message missing required 'role' field")
        if "content" not in msg:
            raise ValueError("Message missing required 'content' field")
        if msg["role"] not in ["system", "user", "assistant"]:
            raise ValueError(
                f"Invalid role '{msg['role']}'. Must be 'system', 'user', or 'assistant'"
            )


def validate_batched_chat_messages(messages):
    """
    Validate format of batched chat messages before API call

    Args:
        messages (List[List[Mapping]]): List of message lists, where each inner list contains
            message dictionaries with 'role' and 'content' fields

    Raises:
        TypeError: If messages format is invalid
        ValueError: If message content is invalid
    """
    if not isinstance(messages, list):
        raise TypeError("Batched messages must be a list")

    if not messages:
        raise ValueError("Batched messages cannot be empty")

    for message_list in messages:
        if not isinstance(message_list, list):
            raise TypeError("Each batch item must be a list of messages")

        # Validate individual messages using existing function
        validate_chat_messages(message_list)


def is_batched_messages(
    messages: Union[List[Mapping], List[List[Mapping]]],
) -> TypeGuard[List[List[Mapping]]]:
    return isinstance(messages, list) and all(isinstance(msg, list) for msg in messages)


def is_simple_messages(
    messages: Union[List[Mapping], List[List[Mapping]]],
) -> TypeGuard[List[Mapping]]:
    return isinstance(messages, list) and all(
        not isinstance(msg, list) for msg in messages
    )


def get_chat_completion(
    model_type: str,
    messages: Union[List[Mapping], List[List[Mapping]]],
    response_format: pydantic.BaseModel = None,
    batched: bool = False,
) -> Union[str, List[str | None]]:
    """
    Generates chat completions using a single model and potentially several messages. Supports closed-source and OSS models.

    Parameters:
        - model_type (str): The type of model to use for generating completions.
        - messages (Union[List[Mapping], List[List[Mapping]]]): The messages to be used for generating completions.
            If batched is True, this should be a list of lists of mappings.
        - response_format (pydantic.BaseModel, optional): The format of the response. Defaults to None.
        - batched (bool, optional): Whether to process messages in batch mode. Defaults to False.
    Returns:
        - str: The generated chat completion(s). If batched is True, returns a list of strings.
    Raises:
        - ValueError: If requested model is not supported by Litellm or TogetherAI.
    """

    # Check for empty messages list
    if not messages or messages == []:
        raise ValueError("Messages cannot be empty")

    # Add validation
    if batched:
        validate_batched_chat_messages(messages)
    else:
        validate_chat_messages(messages)

    if (
        batched
        and is_batched_messages(messages)
        and model_type in TOGETHER_SUPPORTED_MODELS
    ):
        return query_together_api_multiple_calls(
            models=[model_type] * len(messages),
            messages=messages,
            response_formats=[response_format] * len(messages),
        )
    elif (
        batched
        and is_batched_messages(messages)
        and model_type in LITELLM_SUPPORTED_MODELS
    ):
        return query_litellm_api_multiple_calls(
            models=[model_type] * len(messages),
            messages=messages,
            response_formats=[response_format] * len(messages),
        )
    elif (
        not batched
        and is_simple_messages(messages)
        and model_type in TOGETHER_SUPPORTED_MODELS
    ):
        return fetch_together_api_response(
            model=model_type, messages=messages, response_format=response_format
        )
    elif (
        not batched
        and is_simple_messages(messages)
        and model_type in LITELLM_SUPPORTED_MODELS
    ):
        return fetch_litellm_api_response(
            model=model_type, messages=messages, response_format=response_format
        )

    raise ValueError(
        f"Model {model_type} is not supported by Litellm or TogetherAI for chat completions. Please check the model name and try again."
    )


async def aget_chat_completion(
    model_type: str,
    messages: Union[List[Mapping], List[List[Mapping]]],
    response_format: pydantic.BaseModel = None,
    batched: bool = False,
) -> Union[str, List[str | None]]:
    """
    ASYNCHRONOUSLY generates chat completions using a single model and potentially several messages. Supports closed-source and OSS models.

    Parameters:
        - model_type (str): The type of model to use for generating completions.
        - messages (Union[List[Mapping], List[List[Mapping]]]): The messages to be used for generating completions.
            If batched is True, this should be a list of lists of mappings.
        - response_format (pydantic.BaseModel, optional): The format of the response. Defaults to None.
        - batched (bool, optional): Whether to process messages in batch mode. Defaults to False.
    Returns:
        - str: The generated chat completion(s). If batched is True, returns a list of strings.
    Raises:
        - ValueError: If requested model is not supported by Litellm or TogetherAI.
    """

    if batched:
        validate_batched_chat_messages(messages)
    else:
        validate_chat_messages(messages)

    if (
        batched
        and is_batched_messages(messages)
        and model_type in TOGETHER_SUPPORTED_MODELS
    ):
        return await aquery_together_api_multiple_calls(
            models=[model_type] * len(messages),
            messages=messages,
            response_formats=[response_format] * len(messages),
        )
    elif (
        batched
        and is_batched_messages(messages)
        and model_type in LITELLM_SUPPORTED_MODELS
    ):
        return await aquery_litellm_api_multiple_calls(
            models=[model_type] * len(messages),
            messages=messages,
            response_formats=[response_format] * len(messages),
        )
    elif (
        not batched
        and is_simple_messages(messages)
        and model_type in TOGETHER_SUPPORTED_MODELS
    ):
        return await afetch_together_api_response(
            model=model_type, messages=messages, response_format=response_format
        )
    elif (
        not batched
        and is_simple_messages(messages)
        and model_type in LITELLM_SUPPORTED_MODELS
    ):
        return await afetch_litellm_api_response(
            model=model_type, messages=messages, response_format=response_format
        )

    judgeval_logger.error(f"Model {model_type} not supported by either API")
    raise ValueError(
        f"Model {model_type} is not supported by Litellm or TogetherAI for chat completions. Please check the model name and try again."
    )


def get_completion_multiple_models(
    models: List[str],
    messages: List[List[Mapping]],
    response_formats: List[pydantic.BaseModel] | None = None,
) -> List[str | None]:
    """
    Retrieves completions for a single prompt from multiple models in parallel. Supports closed-source and OSS models.

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query. Each inner object corresponds to a single prompt.
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: List of completions from the models in the order of the input models
    Raises:
        ValueError: If a model is not supported by Litellm or Together
    """

    if models is None or models == []:
        raise ValueError("Models list cannot be empty")

    validate_batched_chat_messages(messages)

    if len(models) != len(messages):
        judgeval_logger.error(
            f"Model/message count mismatch: {len(models)} vs {len(messages)}"
        )
        raise ValueError(
            f"Number of models and messages must be the same: {len(models)} != {len(messages)}"
        )
    if response_formats is None:
        response_formats = [None] * len(models)
    # Partition the model requests into TogetherAI and Litellm models, but keep the ordering saved
    together_calls, litellm_calls = {}, {}  # index -> model, message, response_format
    together_responses, litellm_responses = [], []
    for idx, (model, message, r_format) in enumerate(
        zip(models, messages, response_formats)
    ):
        if model in TOGETHER_SUPPORTED_MODELS:
            together_calls[idx] = (model, message, r_format)
        elif model in LITELLM_SUPPORTED_MODELS:
            litellm_calls[idx] = (model, message, r_format)
        else:
            judgeval_logger.error(f"Model {model} not supported by either API")
            raise ValueError(
                f"Model {model} is not supported by Litellm or TogetherAI for chat completions. Please check the model name and try again."
            )

    # Add validation before processing
    for msg_list in messages:
        validate_chat_messages(msg_list)

    # Get the responses from the TogetherAI models
    # List of responses from the TogetherAI models in order of the together_calls dict
    if together_calls:
        together_responses = query_together_api_multiple_calls(
            models=[model for model, _, _ in together_calls.values()],
            messages=[message for _, message, _ in together_calls.values()],
            response_formats=[format for _, _, format in together_calls.values()],
        )

    # Get the responses from the Litellm models
    if litellm_calls:
        litellm_responses = query_litellm_api_multiple_calls(
            models=[model for model, _, _ in litellm_calls.values()],
            messages=[message for _, message, _ in litellm_calls.values()],
            response_formats=[format for _, _, format in litellm_calls.values()],
        )

    # Merge the responses in the order of the original models
    out: List[Union[str, None]] = [None] * len(models)
    for idx, (model, message, r_format) in together_calls.items():
        out[idx] = together_responses.pop(0)
    for idx, (model, message, r_format) in litellm_calls.items():
        out[idx] = litellm_responses.pop(0)
    return out


async def aget_completion_multiple_models(
    models: List[str],
    messages: List[List[Mapping]],
    response_formats: List[pydantic.BaseModel] | None = None,
) -> List[str | None]:
    """
    ASYNCHRONOUSLY retrieves completions for a single prompt from multiple models in parallel. Supports closed-source and OSS models.

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query. Each inner object corresponds to a single prompt.
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: List of completions from the models in the order of the input models
    Raises:
        ValueError: If a model is not supported by Litellm or Together
    """
    if models is None or models == []:
        raise ValueError("Models list cannot be empty")

    if len(models) != len(messages):
        raise ValueError(
            f"Number of models and messages must be the same: {len(models)} != {len(messages)}"
        )
    if response_formats is None:
        response_formats = [None] * len(models)

    validate_batched_chat_messages(messages)

    # Partition the model requests into TogetherAI and Litellm models, but keep the ordering saved
    together_calls, litellm_calls = {}, {}  # index -> model, message, response_format
    together_responses, litellm_responses = [], []
    for idx, (model, message, r_format) in enumerate(
        zip(models, messages, response_formats)
    ):
        if model in TOGETHER_SUPPORTED_MODELS:
            together_calls[idx] = (model, message, r_format)
        elif model in LITELLM_SUPPORTED_MODELS:
            litellm_calls[idx] = (model, message, r_format)
        else:
            raise ValueError(
                f"Model {model} is not supported by Litellm or TogetherAI for chat completions. Please check the model name and try again."
            )

    # Add validation before processing
    for msg_list in messages:
        validate_chat_messages(msg_list)

    # Get the responses from the TogetherAI models
    # List of responses from the TogetherAI models in order of the together_calls dict
    if together_calls:
        together_responses = await aquery_together_api_multiple_calls(
            models=[model for model, _, _ in together_calls.values()],
            messages=[message for _, message, _ in together_calls.values()],
            response_formats=[format for _, _, format in together_calls.values()],
        )

    # Get the responses from the Litellm models
    if litellm_calls:
        litellm_responses = await aquery_litellm_api_multiple_calls(
            models=[model for model, _, _ in litellm_calls.values()],
            messages=[message for _, message, _ in litellm_calls.values()],
            response_formats=[format for _, _, format in litellm_calls.values()],
        )

    # Merge the responses in the order of the original models
    out: List[Union[str, None]] = [None] * len(models)
    for idx, (model, message, r_format) in together_calls.items():
        out[idx] = together_responses.pop(0)
    for idx, (model, message, r_format) in litellm_calls.items():
        out[idx] = litellm_responses.pop(0)
    return out


if __name__ == "__main__":
    batched_messages: List[List[Mapping]] = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
    ]

    non_batched_messages: List[Mapping] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    batched_messages_2: List[List[Mapping]] = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of China?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
    ]

    # Batched
    pprint.pprint(
        get_chat_completion(
            model_type="LLAMA3_405B_INSTRUCT_TURBO",
            messages=batched_messages,
            batched=True,
        )
    )

    # Non batched
    pprint.pprint(
        get_chat_completion(
            model_type="LLAMA3_8B_INSTRUCT_TURBO",
            messages=non_batched_messages,
            batched=False,
        )
    )

    # Batched single completion to multiple models
    pprint.pprint(
        get_completion_multiple_models(
            models=[
                "LLAMA3_70B_INSTRUCT_TURBO",
                "LLAMA3_405B_INSTRUCT_TURBO",
                "gpt-4.1-mini",
            ],
            messages=batched_messages_2,
        )
    )

ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]
OptExcInfo: TypeAlias = ExcInfo | tuple[None, None, None]
