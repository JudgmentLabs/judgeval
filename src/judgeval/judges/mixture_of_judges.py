"""
Implementation for Mixture of Judges model through Judgeval

Enables client to use multiple models to generate responses and then aggregate them into a single response.
"""

import pydantic
from typing import List, Union
from judgeval.judges import JudgevalJudge
from judgeval.common.utils import (
    get_completion_multiple_models,
    get_chat_completion,
    aget_completion_multiple_models,
    aget_chat_completion,
)
from judgeval.common.logger import judgeval_logger


def build_dynamic_mixture_prompt(
    judge_responses: List[str],
    custom_system_prompt: str | None = None,
    custom_conversation_history: List[dict] | None = None,
) -> List[dict]:
    """
    Dynamically builds a prompt to mix judge responses together for the Mixture of Judges model.

    In this implementation, we simply concatenate the judge responses into a formatted string, then
    pass it into a default prompt template. This template can be customized by providing a custom prompt.

    Args:
        judge_responses (List[str]): List of responses from individual judges to be synthesized
        custom_system_prompt (str, optional): Custom system prompt to override the default one. Defaults to None.
        custom_conversation_history (List[dict], optional): Custom conversation history to override the default one. Defaults to None.
    """
    formatted_responses = "\n".join(
        [
            f"# Judge {i + 1}'s response: #\n{response}"
            for i, response in enumerate(judge_responses)
        ]
    )

    # This is the default prompt for the Mixture of Judges model
    """
    You are tasked with synthesizing responses from multiple expert judges. You will receive N individual answers on the same topic. Your job is to:

    1. Analyze and compare the key points, patterns, and agreements between the answers.
    2. Identify the consensus by focusing on areas where most or all of the answers align. Consider common reasoning and frequently mentioned conclusions.
    3. Condense the responses into a single, coherent, and concise answer that represents the collective judgment of the group.
    4. When opinions differ or contradict, highlight the most supported viewpoint while briefly acknowledging the dissenting perspectives.
    5. Ensure the final answer is balanced and clear, providing a comprehensive summary that captures the wisdom of all judges while avoiding repetition.

    ## Start of Judge Responses ##
    {{judge_responses}}
    ## End of Judge Responses ##
    Synthesized response:
    """

    default_conversation = [  # inject the judge responses into the default prompt
        {
            "role": "system",
            "content": "You are tasked with synthesizing responses from multiple expert judges. You will receive N individual answers on the same topic. Your job is to:\n1. Analyze and compare the key points, patterns, and agreements between the answers.\n2. Identify the consensus by focusing on areas where most or all of the answers align. Consider common reasoning and frequently mentioned conclusions.\n3. Condense the responses into a single, coherent, and concise answer that represents the collective judgment of the group.\n4. When opinions differ or contradict, highlight the most supported viewpoint while briefly acknowledging the dissenting perspectives.\n5. Ensure the final answer is balanced and clear, providing a comprehensive summary that captures the wisdom of all judges while avoiding repetition.\n\n**IMPORTANT**: IF THE JUDGE RESPONSES ARE IN JSON FORMAT, YOU MUST RESPOND USING THE SAME JSON FORMAT THAT THE RESPONSES ARE IN. If the judge responses are in JSON, you MUST RESPOND IN VALID JSON FORMAT. ",
        },
        {
            "role": "user",
            "content": '## Start of Judge Responses ## \n# Judge 1\'s response: #\n{\n"claims": [\n{\n"claim": "A 30-day full refund is offered.",\n"quote": "We offer a 30-day full refund at no extra cost."\n},\n{\n"claim": "The 30-day full refund comes at no extra cost.",\n"quote": "We offer a 30-day full refund at no extra cost."\n}\n]\n}\n\n# Judge 2\'s response: #\n{\n    "claims": [\n        {\n            "claim": "A full refund is offered within 30 days.",\n            "quote": "We offer a 30-day full refund at no extra cost."\n        },\n        {\n            "claim": "The 30-day full refund is offered at no extra cost.",\n            "quote": "We offer a 30-day full refund at no extra cost."\n        }\n    ]\n}\n# Judge 3\'s response: #\n {\n    "claims": [\n        {\n            "claim": "A 30-day full refund is offered.",\n            "quote": "We offer a 30-day full refund at no extra cost."\n        },\n        {\n            "claim": "The 30-day full refund is offered at no extra cost.",\n            "quote": "We offer a 30-day full refund at no extra cost."\n        }\n    ]\n}\n## End of Judge Responses ##\nSynthesized response:',
        },
        {
            "role": "assistant",
            "content": 'The consensus among the judges is clear and unanimous. All three judges agree that a 30-day full refund is offered, and this refund is available at no extra cost. This conclusion is consistently supported by their statements, as each of their claims is directly quoted as: "We offer a 30-day full refund at no extra cost." There are no dissenting perspectives or opposing views provided in any of the responses, indicating complete alignment on this topic.\n\nJSON:\n{\n    "claims": [\n        {\n            "claim": "A full refund is offered within 30 days.",\n            "quote": "We offer a 30-day full refund at no extra cost."\n        },\n        {\n            "claim": "The 30-day full refund is offered at no extra cost.",\n            "quote": "We offer a 30-day full refund at no extra cost."\n        }\n    ]\n}',
        },
        {
            "role": "user",
            "content": "## Start of Judge Responses ##\n# Judge 1's response: # \nThe capital of France is Paris.\n\n# Judge 2's response: #\nThe capital of France is Paris.\n\n# Judge 3's response: # \nThe capital of France is Paris. It's one of the most popular tourist destinations in the world, known for its art, culture, and history. It's also famous for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.\n\n## End of Judge Responses ##\nSynthesized response:",
        },
        {
            "role": "assistant",
            "content": "The capital of France is Paris. It is widely recognized as one of the world's most popular tourist destinations, celebrated for its rich art, culture, and history. Paris is renowned for its iconic landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
        },
        {
            "role": "user",
            "content": f"## Start of Judge Responses ##\n{formatted_responses}\n## End of Judge Responses ##\nSynthesized response:\n",
        },
    ]

    # If a custom system prompt is provided, validate and use it
    if custom_system_prompt is not None:
        if not isinstance(custom_system_prompt, str):
            judgeval_logger.error(
                f"TypeError: Custom system prompt must be a string. Received: {type(custom_system_prompt)}."
            )
            raise TypeError(
                f"Custom system prompt must be a string. Received: {type(custom_system_prompt)}."
            )
        if not custom_system_prompt:
            raise ValueError("Custom system prompt cannot be empty")
        # Override the default system prompt, but also add special instructions for handling JSON
        default_conversation[0]["content"] = (
            custom_system_prompt
            + "\n\n**IMPORTANT**: IF THE JUDGE RESPONSES ARE IN JSON FORMAT, YOU MUST RESPOND USING THE SAME JSON FORMAT THAT THE RESPONSES ARE IN. If the judge responses are in JSON, you MUST RESPOND IN VALID JSON FORMAT."
        )

    # If a custom conversation history is provided, append the judge responses to it
    if custom_conversation_history is not None:
        # Validate custom conversation history format
        for message in custom_conversation_history:
            if not isinstance(message, dict):
                raise TypeError(
                    f"Custom conversation history must be a list of dictionaries. Received: {message}."
                )

            if "role" not in message or "content" not in message:
                raise ValueError("Each message must have 'role' and 'content' keys")

            if not isinstance(message["role"], str) or not isinstance(
                message["content"], str
            ):
                raise TypeError(
                    f"Message role and content must be strings. Received: {type(message['role'])}, {type(message['content'])}."
                )

            if message["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    f"Message role must be one of: 'system', 'user', 'assistant'. Received: {message['role']}."
                )

        judge_responses_prompt = {
            "role": "user",
            "content": f"## Start of Judge Responses ##\n{formatted_responses}\n## End of Judge Responses ##\nSynthesized response:\n",
        }
        return custom_conversation_history + [judge_responses_prompt]

    # Otherwise return the default conversation with system prompt and examples
    # No customization, return the default conversation with system prompt and examples
    return default_conversation


BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
]  # for string inputs, we need to add the user query to a base conversation, since LiteLLM only accepts a list of dictionaries as a chat history


class MixtureOfJudges(JudgevalJudge):
    """
    IMPORTANT: When supplying custom prompts and conversation histories for aggregation, supply them in the following format:
    in kwargs:
    {
        "custom_prompt": "Your custom prompt here",
        "custom_conversation": [
            {"role": "system", "content": "System message 1"},
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant message 1"},
            ...
        ]
    }
    """

    def __init__(
        self,
        models: List[str] = [
            "QWEN",
            "LLAMA3_70B_INSTRUCT_TURBO",
            "MISTRAL_8x22B_INSTRUCT",
        ],
        aggregator: str = "gpt-4.1",
        **kwargs,
    ):
        """
        `models` are the individual judge models to be used for generating responses.
        `aggregator` is the model that will aggregate the responses from the individual judges.

        kwargs include "custom_prompt" and "custom_conversation" for customizing the prompt for the Mixture of Judges model.
        """
        self.models = models
        self.aggregator = aggregator
        self.kwargs = kwargs
        super().__init__(model_name=models)

    def generate(
        self,
        input: Union[str, List[dict]],
        response_schema: pydantic.BaseModel = None,
        aggregation_schema: pydantic.BaseModel = None,
        **kwargs,
    ) -> str:
        """
        Args:
            input (Union[str, List[Mapping[str, str]]]): Input query or conversation history to the model.
            response_schema (pydantic.BaseModel): Response schema for individual judge models.
            aggregation_schema (pydantic.BaseModel): Response schema for the aggregator model.
            kwargs: Additional keyword arguments.
        """

        # Convert input to conversation format if needed
        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
        elif isinstance(input, list):
            convo = input
        else:
            judgeval_logger.error(f"Invalid input type received: {type(input)}")
            raise TypeError(
                f"Input must be a string or a list of dictionaries. Input type of: {type(input)}"
            )

        try:
            responses = get_completion_multiple_models(
                models=self.models,
                messages=[convo] * len(self.models),
                response_formats=[response_schema] * len(self.models),
            )
        except Exception:
            raise

        compiled_mixture_prompt = build_dynamic_mixture_prompt(
            responses,
            self.kwargs.get("custom_prompt"),
            self.kwargs.get("custom_conversation"),
        )

        try:
            mixed_response = get_chat_completion(
                model_type=self.aggregator,
                messages=compiled_mixture_prompt,
                response_format=aggregation_schema,
            )
        except Exception:
            raise

        return mixed_response

    async def a_generate(
        self,
        input: Union[str, List[dict]],
        response_schema: pydantic.BaseModel = None,
        aggregation_schema: pydantic.BaseModel = None,
        **kwargs,
    ) -> str:
        """
        Args:
            input (Union[str, List[Mapping[str, str]]]): Input query or conversation history to the model.
            response_schema (pydantic.BaseModel): Response schema for individual judge models.
            aggregation_schema (pydantic.BaseModel): Response schema for the aggregator model.
            kwargs: Additional keyword arguments.
        """

        # Convert input to conversation format if needed
        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
        elif isinstance(input, list):
            convo = input
        else:
            judgeval_logger.error(f"Invalid input type received: {type(input)}")
            raise TypeError(
                f"Input must be a string or a list of dictionaries. Input type of: {type(input)}"
            )

        try:
            responses = await aget_completion_multiple_models(
                models=self.models,
                messages=[convo] * len(self.models),
                response_formats=[response_schema] * len(self.models),
            )
        except Exception:
            raise

        compiled_mixture_prompt = build_dynamic_mixture_prompt(
            responses,
            self.kwargs.get("custom_prompt"),
            self.kwargs.get("custom_conversation"),
        )

        try:
            mixed_response = await aget_chat_completion(
                model_type=self.aggregator,
                messages=compiled_mixture_prompt,
                response_format=aggregation_schema,
            )
        except Exception:
            raise

        return mixed_response

    def load_model(self):
        return self.models

    def get_model_name(self) -> List[str]:
        return self.models
