import pydantic
from typing import List, Union, Mapping

from judgeval.judges import JudgevalJudge
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL
from judgeval.judges.todo import (
    fetch_litellm_api_response,
    afetch_litellm_api_response,
)

BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
]


class LiteLLMJudge(JudgevalJudge):
    def __init__(self, model: str = JUDGMENT_DEFAULT_GPT_MODEL, **kwargs):
        self.model = model
        self.kwargs = kwargs
        super().__init__(model_name=model)

    def generate(
        self,
        input: Union[str, List[Mapping[str, str]]],
        schema: Union[pydantic.BaseModel, None] = None,
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
            raise TypeError(
                f"Input must be a string or a list of dictionaries. Input type of: {type(input)}"
            )

    async def a_generate(
        self,
        input: Union[str, List[Mapping[str, str]]],
        schema: Union[pydantic.BaseModel, None] = None,
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
            raise TypeError(f"Input must be a string or a list of dictionaries. Input type of: {type(input)}")  # type: ignore[unreachable]

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return self.model
