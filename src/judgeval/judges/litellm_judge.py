import pydantic
from typing import Dict, List, Union, Mapping, Any

from judgeval.constants import ACCEPTABLE_MODELS
from judgeval.judges import JudgevalJudge
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL

try:
    import litellm
except ImportError:
    raise ImportError(
        "Litellm is not installed and required for the litellm judge. Please install it with `pip install litellm`."
    )


def fetch_litellm_api_response(
    model: str,
    messages: List[Dict[str, str]],
    response_format: Union[Dict[str, Any], None] = None,
) -> str:
    if response_format is not None:
        response = litellm.completion(
            model=model,
            messages=messages,
            response_format=response_format,
        )
    else:
        response = litellm.completion(
            model=model,
            messages=messages,
        )

    content = response.choices[0].message.content  # type: ignore[attr-defined]
    if content is None:
        raise ValueError("Received empty response from litellm")
    return content


async def afetch_litellm_api_response(
    model: str,
    messages: List[Dict[str, str]],
    response_format: Union[Dict[str, Any], None] = None,
) -> str:
    if not messages:
        raise ValueError("Messages cannot be empty")

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

    content = response.choices[0].message.content  # type: ignore[attr-defined]
    if content is None:
        raise ValueError("Received empty response from litellm")
    return content


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
        response_format = schema.model_json_schema() if schema else None

        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            return fetch_litellm_api_response(
                model=self.model, messages=convo, response_format=response_format
            )
        elif isinstance(input, list):
            messages = [dict(msg) for msg in input]
            return fetch_litellm_api_response(
                model=self.model, messages=messages, response_format=response_format
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
        response_format = schema.model_json_schema() if schema else None

        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            response = await afetch_litellm_api_response(
                model=self.model, messages=convo, response_format=response_format
            )
            return response
        elif isinstance(input, list):
            messages = [dict(msg) for msg in input]
            response = await afetch_litellm_api_response(
                model=self.model, messages=messages, response_format=response_format
            )
            return response
        else:
            raise TypeError(
                f"Input must be a string or a list of dictionaries. Input type of: {type(input)}"
            )

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return self.model
