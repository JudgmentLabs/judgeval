"""
Implementation of using TogetherAI inference for judges.
"""

from pydantic import BaseModel
from typing import Dict, List, Union, Any, cast
from judgeval.judges import JudgevalJudge
from judgeval.logger import judgeval_logger
from judgeval.env import (
    JUDGMENT_DEFAULT_TOGETHER_MODEL,
    TOGETHERAI_API_KEY,
    TOGETHER_API_KEY,
)

together_api_key = TOGETHERAI_API_KEY or TOGETHER_API_KEY
if together_api_key:
    try:
        from together import Together, AsyncTogether  # type: ignore[import-untyped]

        together_client = Together(api_key=together_api_key)
        async_together_client = AsyncTogether(api_key=together_api_key)
    except Exception:
        pass


def fetch_together_api_response(
    model: str,
    messages: List[Dict[str, str]],
    response_format: Union[Dict[str, Any], None] = None,
) -> str:
    if not messages:
        raise ValueError("Messages cannot be empty")

    if response_format is not None:
        response = together_client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
        )
    else:
        response = together_client.chat.completions.create(
            model=model,
            messages=messages,
        )

    content = response.choices[0].message.content  # type: ignore[attr-defined]
    if content is None:
        raise ValueError("Received empty response from TogetherAI")
    return cast(str, content)


async def afetch_together_api_response(
    model: str,
    messages: List[Dict[str, str]],
    response_format: Union[Dict[str, Any], None] = None,
) -> str:
    if not messages:
        raise ValueError("Messages cannot be empty")

    if response_format is not None:
        response = await async_together_client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
        )
    else:
        response = await async_together_client.chat.completions.create(
            model=model,
            messages=messages,
        )

    content = response.choices[0].message.content  # type: ignore[attr-defined]
    if content is None:
        raise ValueError("Received empty response from TogetherAI")
    return cast(str, content)


BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
]


class TogetherJudge(JudgevalJudge):
    def __init__(self, model: str = JUDGMENT_DEFAULT_TOGETHER_MODEL, **kwargs):
        self.model = model
        self.kwargs = kwargs
        super().__init__(model_name=model)

    def generate(
        self,
        input: Union[str, List[Dict[str, str]]],
        schema: Union[BaseModel, None] = None,
    ) -> str:
        response_format = schema.model_json_schema() if schema else None

        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            return fetch_together_api_response(
                self.model, convo, response_format=response_format
            )
        elif isinstance(input, list):
            messages = [dict(msg) for msg in input]
            return fetch_together_api_response(
                self.model, messages, response_format=response_format
            )
        else:
            judgeval_logger.error(f"Invalid input type received: {type(input)}")
            raise TypeError("Input must be a string or a list of dictionaries.")

    async def a_generate(
        self,
        input: Union[str, List[Dict[str, str]]],
        schema: Union[BaseModel, None] = None,
    ) -> str:
        response_format = schema.model_json_schema() if schema else None

        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            res = await afetch_together_api_response(
                self.model, convo, response_format=response_format
            )
            return res
        elif isinstance(input, list):
            messages = [dict(msg) for msg in input]
            res = await afetch_together_api_response(
                self.model, messages, response_format=response_format
            )
            return res
        else:
            raise TypeError("Input must be a string or a list of dictionaries.")

    def load_model(self) -> str:
        return self.model

    def get_model_name(self) -> str:
        return self.model
