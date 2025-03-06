"""
Code that implements a prompt-based scorer for evaluating examples.

The PromptScorer class is a base class that can be used to create custom scoring metrics using LLM prompts.
To implement a subclass of PromptScorer, you need to implement the following methods:
- build_measure_prompt(): builds the conversation prompt that is sent to the LLM judge
- build_schema(): defines the expected response schema from the LLM
- process_response(): parses the response from the LLM judge
- success_check(): determines whether the evaluation was successful

The core idea of PromptScorer is to provide a flexible way to create custom scoring metrics
by leveraging LLM judges to evaluate examples. The scorer constructs a prompt, sends it to 
the judge, and parses the structured response to determine a score.

For example, the SentimentScorer subclass uses PromptScorer to detect negative sentiment in responses
by prompting an LLM to rate the negativity on a 1-5 scale and provide a reason for the rating.

The PromptScorer supports both synchronous and asynchronous evaluation modes, includes optional
reason fields in responses, and can operate in strict mode with higher thresholds.

NOTE: When implementing build_measure_prompt and build_schema:
- The prompt should guide the LLM to generate a response matching your schema
- The schema should include "score" and optionally "reason" fields
- The score field type and range should match your scoring criteria
- The reason field provides explanatory context for the score
"""

from abc import abstractmethod
from typing import List, Optional, Tuple, Any, Mapping
from pydantic import BaseModel, model_serializer, Field

from judgeval.data import Example
from judgeval.scorers import JudgevalScorer
from judgeval.scorers.utils import (
    scorer_progress_meter, 
    parse_response_json,
    get_or_create_event_loop,
    create_verbose_logs
)


class ReasonScore(BaseModel):
    reason: str
    score: float


class PromptScorer(JudgevalScorer, BaseModel):
    name: str
    score_type: str
    threshold: float = Field(default=0.5)
    using_native_model: bool = Field(default=True)

    # DO NOT SET THESE FIELDS MANUALLY, THEY ARE SET BY THE SCORE_EXAMPLE METHOD
    _response: Optional[dict] = None
    _result: Optional[float] = None
    
    def __init__(
        self,
        name: str, 
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        # Initialize BaseModel first
        BaseModel.__init__(
            self,
            name=name,
            score_type=name,
            threshold=1 if strict_mode else threshold,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
        )
        # Then initialize JudgevalScorer
        JudgevalScorer.__init__(
            self,
            score_type=name,
            threshold=1 if strict_mode else threshold,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
        )

    def score_example(
            self, 
            example: Example, 
            _show_indicator: bool = True
            ) -> float:
        """
        Synchronous method for scoring an example using the prompt criteria.
        """
        with scorer_progress_meter(self, display_meter=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_score_example(example, _show_indicator=False)
                )
            else:
                result, reason = self.evaluate(example)
                self.reason = reason
                self._result = result
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Results: {self._result}\nReason: {self.reason}",
                    ],
                )
                return result

    async def a_score_example(
            self,
            example: Example,
            _show_indicator: bool = True,
            ) -> float: 
        """
        Async method for scoring an example using the prompt criteria.
        """
        with scorer_progress_meter(self, display_meter=_show_indicator):
            result, reason = await self.a_evaluate(example)
            self.reason = reason
            self._result = result
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Results: {self._result}\nReason: {self.reason}",
                ],
            )
            return result
    
    def evaluate(self, example: Example) -> Tuple[Any, str]:
        """
        Synchronous helper method for evaluating an example using the prompt criteria.

        Builds a custom prompt using `build_measure_prompt` and sends it to the judge model 
        for evaluation. The result is then parsed as JSON and returned.

        NOTE: It is assumed that the model response will be JSON and contain a "score" and "reason" field.
        """
        prompt = self._build_measure_prompt(example)
        if self.using_native_model:
            res = self.model.generate(prompt)
            response = parse_response_json(res, self)
            result, reason = self._process_response(response)
            return result, reason
        else:
            raise NotImplementedError("Non-native judge models are not supported in synchronous mode yet.")

    async def a_evaluate(self, example: Example) -> Tuple[Any, str]:
        """
        Asynchronous helper method for evaluating an example using the prompt criteria.

        Builds a custom prompt using `build_measure_prompt` and sends it to the judge model 
        for evaluation. The result is then parsed as JSON and returned.

        NOTE: It is assumed that the model response will be JSON and contain a "score" and "reason" field.
        """
        judge_prompt = self._build_measure_prompt(example)
        schema = self._build_schema()
        prompt = self._enforce_prompt_format(judge_prompt=judge_prompt, schema=schema)
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            response = parse_response_json(res, self)
            self._response = response

            result, reason = self._process_response(response)
            self.score = result
            self.reason = reason
            self._response = response
            return result, reason
        else:
            raise NotImplementedError("Non-native judge models are not supported in async mode yet.")

    # TODO: can we make this take *args and **kwargs? How does that work with a_evaluate() since we'd have to pass the same args
    @abstractmethod
    def _build_measure_prompt(self, example: Example) -> List[dict]:
        # builds the prompt that is sent to the model inside of the `score_example()` method
        # returns either a string prompt or a conversation prompt of the form [{"role": "system", "content": "..."}, ...]

        """
        This function creates the prompt that the judge model uses to evaluate examples.

        The prompt is typically a set of instructions that the judge model uses to evaluate the example.

        This function returns a conversation prompt of the form 
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        A basic version of implementing this function could be as follows:
        SYSTEM_ROLE = ...
        return [
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": f"Response: {example.actual_output}\n\nYour judgment: "}
        ]
        """
        pass
    
    # TODO: does this need to take *args and **kwargs? How does that work with a_evaluate() since we'd have to pass the same args
    @abstractmethod
    def _build_schema(self) -> dict:
        """
        This function returns a dictionary that represents the schema of the JSON response that the judge model should return.

        The keys of the dictionary are the expected keys in the response, and the values are the types of the corresponding values.

        Example: If you want to have the judge model return a score and a reason, you would write:
        return {"score": int, "reason": str}
        """
        pass
    
    def _enforce_prompt_format(self, judge_prompt: List[dict], schema: dict):
        """
        Formats the final prompt to the judge model.

        This function takes a list of dictionaries (`judge_prompt`) and a schema dictionary (`schema`), 
        and appends a schema enforcement prompt to the content of the first dictionary in the list, which is assumed to be the system prompt. 
        The schema enforcement prompt instructs the judge model to provide its response in a specific JSON format.

        Args:
            judge_prompt (List[dict]): A list of dictionaries representing the judge prompt. 
                                       Each dictionary should contain a "content" key.
            schema (dict): A dictionary representing the schema. The keys are the expected keys in the response, 
                           and the values are the types of the corresponding values.

        Returns:
            List[dict]: The modified judge prompt with the schema enforcement prompt appended to the content 
                        of the first dictionary.

        Raises:
            TypeError: If `judge_prompt` is not a list of dictionaries.

        Example:
            judge_prompt = [{"content": "Please evaluate the following:"}]
            schema = {"score": int, "comments": str}
            formatted_prompt = format_measure_prompt(judge_prompt, schema)
            # formatted_prompt[0]["content"] will include the schema enforcement prompt
        """
        SCHEMA_ENFORCEMENT_PROMPT = "\n\nPlease provide your response in the following JSON format: {"
        if isinstance(judge_prompt, list) and all(isinstance(item, dict) for item in judge_prompt):
            # create formatting string for schema enforcement
            # schema is a map between key and type of the value 
            for key, key_type in schema.items():
                SCHEMA_ENFORCEMENT_PROMPT += f'"{key}": <{key}> ({key_type.__name__}), '
            SCHEMA_ENFORCEMENT_PROMPT = SCHEMA_ENFORCEMENT_PROMPT[:-2] + "}"  # remove trailing comma and space
            judge_prompt[0]["content"] += SCHEMA_ENFORCEMENT_PROMPT
            return judge_prompt
        else:
            raise TypeError(f"Prompt must be a list of dictionaries. Got {type(judge_prompt)} instead.")

    @abstractmethod 
    def _process_response(self, response: dict):
        """
        Customizable method for processing the response from the judge model.

        You can add any additional logic to parse the JSON response here and return the result and reason for decision.

        If you don't need a reason for the decision, you can simply return (score, None).

        Example:
        score = response["score"]
        reason = response["reason"]
        return score, reason
        """
        pass

    @abstractmethod
    def _success_check(self, **kwargs) -> bool:
        """
        Determines whether or not the PromptScorer should consider the evaluation of a single example successful.
        """
        pass
    
    @property
    def __name__(self):
        return self.name


class ClassifierScorer(PromptScorer):

    """
    This is a PromptScorer that takes 
    1. a system role that may involve the Example object
    2. options for scores on the example
    
    and uses a judge to execute the evaluation from the system role and classify into one of the options

    ex:
    system_role = "You are a judge that evaluates whether the response is positive or negative. The response is: {example.actual_output}"
    options = {"positive": 1, "negative": 0}
    """

    conversation: List[dict]
    options: Mapping[str, float]
    
    def __init__(self, name: str, slug: str, conversation: List[dict], options: Mapping[str, float], 
                 threshold: float = 0.5, include_reason: bool = True, 
                 async_mode: bool = True, strict_mode: bool = False, verbose_mode: bool = False):
        # Initialize BaseModel first with all fields
        BaseModel.__init__(
            self,
            name=name,
            slug=slug,
            score_type=name,
            conversation=conversation,
            options=options,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
        )
        # Then initialize JudgevalScorer
        JudgevalScorer.__init__(
            self,
            score_type=name,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
        )

    def _build_measure_prompt(self, example: Example) -> List[dict]:
        """
        Builds the measure prompt for the classifier scorer.

        Args:
            example (Example): The example to build the prompt for

        Returns:
            List[dict]: The measure prompt for the classifier scorer
        """
        replacement_words = {
            "{{actual_output}}": example.actual_output,
            "{{expected_output}}": example.expected_output,
            "{{context}}": example.context,
            "{{retrieval_context}}": example.retrieval_context,
            "{{tools_called}}": example.tools_called,
            "{{expected_tools}}": example.expected_tools,
        }
        # Make a copy of the conversation to avoid modifying the original
        conversation_copy = [dict(message) for message in self.conversation]
        
        # Only replace if double brackets are found in the content
        for message in conversation_copy:
            content = message["content"]
            if "{{" in content:
                for key, value in replacement_words.items():
                    if key in content:
                        message["content"] = content.replace(key, str(value))
        return conversation_copy

    def _build_schema(self) -> dict:
        return self.options
    
    def _enforce_prompt_format(self, judge_prompt: List[dict], schema: dict) -> List[dict]:
        """
        Enforces the judge model to choose an option from the schema.

        We want the model to choose an option from the schema and a reason for the choice.
        """
        options = list(schema.keys())
        options_str = ", ".join(options)
        
        system_role = judge_prompt[0]["content"]
        system_role += (
            f"\n\nYou must choose one of the following options: {options_str}. "
            "Format your response as a JSON object with two fields:\n"
            "1. 'choice': Your selected option (must be one of the provided choices)\n"
            "2. 'reason': A brief explanation for why you made this choice\n\n"
            "Example response format:\n"
            "{\n"
            '    "choice": "<one of the valid options>",\n'
            '    "reason": "<your explanation>"\n'
            "}"
        )
        
        judge_prompt[0]["content"] = system_role
        return judge_prompt

    def _process_response(self, response: dict) -> Tuple[float, str]:
        choice = response.get("choice")
        if choice not in self.options:
            raise ValueError(f"Invalid choice: {choice}. Expected one of: {self.options.keys()}")
        reason = response.get("reason", "No reason could be found in model response.")
        return self.options[choice], reason

    def _success_check(self, **kwargs) -> bool:
        return self.score >= self.threshold
    
    def update_name(self, name: str):
        """
        Updates the name of the scorer.
        """
        self.name = name
        
    def update_threshold(self, threshold: float):
        """
        Updates the threshold of the scorer.
        """
        self.threshold = threshold
    
    def update_conversation(self, conversation: List[dict]):
        """
        Updates the conversation with the new conversation.
        
        Sample conversation:
        [{'role': 'system', 'content': "Did the chatbot answer the user's question in a kind way?: {{actual_output}}."}]
        """
        self.conversation = conversation
        
    def update_options(self, options: Mapping[str, float]):
        """
        Updates the options with the new options.
        
        Sample options:
        {"yes": 1, "no": 0}
        """
        self.options = options

    def __str__(self):
        return f"ClassifierScorer(name={self.name}, slug={self.slug}, conversation={self.conversation}, threshold={self.threshold}, options={self.options})"

    @model_serializer
    def serialize_model(self) -> dict:
        """
        Defines how the ClassifierScorer should be serialized when model_dump() is called.
        """
        return {
            "name": self.name,
            "score_type": self.score_type,
            "conversation": self.conversation,
            "options": self.options,
            "threshold": self.threshold,
            "include_reason": self.include_reason,
            "async_mode": self.async_mode,
            "strict_mode": self.strict_mode,
            "verbose_mode": self.verbose_mode,
        }
