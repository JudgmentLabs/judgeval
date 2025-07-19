import pydantic
from typing import List, Union, Mapping

from judgeval.judges import JudgevalJudge
from judgeval.common.utils import (
    afetch_litellm_api_response,
    fetch_litellm_api_response,
)
from judgeval.common.logger import judgeval_logger

BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
]  # for string inputs, we need to add the user query to a base conversation, since LiteLLM only accepts a list of dictionaries as a chat history


class LiteLLMJudge(JudgevalJudge):
    def __init__(self, model: str = "gpt-4.1-mini", **kwargs):
        self.model = model
        self.kwargs = kwargs
        super().__init__(model_name=model)

    def generate(
        self,
        input: Union[str, List[Mapping[str, str]]],
        schema: pydantic.BaseModel = None,
    ) -> str:
        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            return fetch_litellm_api_response(
                model=self.model, messages=convo, response_format=schema
            )
        elif isinstance(input, list):
            return fetch_litellm_api_response(
                model=self.model, messages=input, response_format=schema
            )
        else:
            judgeval_logger.error(f"Invalid input type received: {type(input)}")
            raise TypeError(
                f"Input must be a string or a list of dictionaries. Input type of: {type(input)}"
            )

    async def a_generate(
        self,
        input: Union[str, List[Mapping[str, str]]],
        schema: pydantic.BaseModel = None,
    ) -> str:
        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            response = await afetch_litellm_api_response(
                model=self.model, messages=convo, response_format=schema
            )
            return response
        elif isinstance(input, list):
            response = await afetch_litellm_api_response(
                model=self.model, messages=input, response_format=schema
            )
            return response
        else:
            judgeval_logger.error(f"Invalid input type received: {type(input)}")
            raise TypeError(
                f"Input must be a string or a list of dictionaries. Input type of: {type(input)}"
            )

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return self.model
