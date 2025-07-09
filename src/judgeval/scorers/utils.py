"""
Util functions for Scorer objects
"""

import asyncio
import nest_asyncio
import inspect
import json
import re
from typing import List, Optional

from judgeval.scorers import BaseScorer
from judgeval.data import Example, ExampleParams
from judgeval.scorers.exceptions import MissingExampleParamsError


def clone_scorers(scorers: List[BaseScorer]) -> List[BaseScorer]:
    """
    Creates duplicates of the scorers passed as argument.
    """
    cloned_scorers = []
    for s in scorers:
        scorer_class = type(s)
        args = vars(s)

        signature = inspect.signature(scorer_class.__init__)
        valid_params = signature.parameters.keys()
        valid_args = {key: args[key] for key in valid_params if key in args}

        cloned_scorer = scorer_class(**valid_args)
        # kinda hacky, but in case the class inheriting from BaseScorer doesn't have `model` in its __init__,
        # we need to explicitly include it here so that we can add the judge model to the cloned scorer
        cloned_scorer._add_model(model=args.get("model"))
        cloned_scorers.append(cloned_scorer)
    return cloned_scorers


def parse_response_json(llm_response: str, scorer: Optional[BaseScorer] = None) -> dict:
    """
    Extracts JSON output from an LLM response and returns it as a dictionary.

    If the JSON is invalid, the error is forwarded to the `scorer`, if provided.

    Args:
        llm_response (str): The response from an LLM.
        scorer (BaseScorer, optional): The scorer object to forward errors to (if any).
    """
    start = llm_response.find("{")  # opening bracket
    end = llm_response.rfind("}") + 1  # closing bracket

    if end == 0 and start != -1:  # add the closing bracket if it's missing
        llm_response = llm_response + "}"
        end = len(llm_response)

    json_str = (
        llm_response[start:end] if start != -1 and end != 0 else ""
    )  # extract the JSON string
    json_str = re.sub(
        r",\s*([\]}])", r"\1", json_str
    )  # Remove trailing comma if present

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a stronger evaluation model."
        if scorer is not None:
            scorer.error = error_str
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create an asyncio event loop.

    This function attempts to retrieve the current event loop using `asyncio.get_event_loop()`.
    If the event loop is already running, it applies the `nest_asyncio` patch to allow nested
    asynchronous execution. If the event loop is closed or not found, it creates a new event loop
    and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.

    Raises:
        RuntimeError: If the event loop is closed.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print(
                "Event loop is already running. Applying nest_asyncio patch to allow async execution..."
            )
            nest_asyncio.apply()

        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def check_example_params(
    example: Example,
    example_params: List[ExampleParams],
    scorer: BaseScorer,
):
    if isinstance(example, Example) is False:
        error_str = f"in check_example_params(): Expected example to be of type 'Example', but got {type(example)}"
        scorer.error = error_str
        raise MissingExampleParamsError(error_str)

    missing_params = []
    for param in example_params:
        if getattr(example, param.value) is None:
            missing_params.append(f"'{param.value}'")

    if missing_params:
        if len(missing_params) == 1:
            missing_params_str = missing_params[0]
        elif len(missing_params) == 2:
            missing_params_str = " and ".join(missing_params)
        else:
            missing_params_str = (
                ", ".join(missing_params[:-1]) + ", and " + missing_params[-1]
            )

        error_str = f"{missing_params_str} fields in example cannot be None for the '{scorer.__name__}' scorer"
        scorer.error = error_str
        raise MissingExampleParamsError(error_str)
