"""
This module contains utility functions for judge models.
"""

import litellm
from typing import Optional, Union, Tuple

from judgeval.exceptions import InvalidJudgeModelError
from judgeval.judges import JudgevalJudge, LiteLLMJudge, TogetherJudge
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL
from judgeval.constants import (
    TOGETHER_SUPPORTED_MODELS,
    JUDGMENT_SUPPORTED_MODELS,
)

LITELLM_SUPPORTED_MODELS = set(litellm.model_list)


def create_judge(
    model: Optional[Union[str, JudgevalJudge]] = None,
) -> Tuple[JudgevalJudge, bool]:
    """
    Creates a judge model from string(s) or a judgeval judge object.

    If `model` is a single string, it is assumed to be a judge model name.
    If `model` is a list of strings, it is assumed to be a list of judge model names (for MixtureOfJudges).
    If `model` is a judgeval judge object, it is returned as is.

    Returns a tuple of (initialized judgevalBaseLLM, using_native_model boolean)
    If no model is provided, uses GPT4o as the default judge.
    """
    if model is None:  # default option
        return LiteLLMJudge(model=JUDGMENT_DEFAULT_GPT_MODEL), True
    if not isinstance(model, (str, list, JudgevalJudge)):
        raise InvalidJudgeModelError(
            f"Model must be a string, list of strings, or a judgeval judge object. Got: {type(model)} instead."
        )
    # If model is already a valid judge type, return it and mark native
    if isinstance(model, (JudgevalJudge, LiteLLMJudge, TogetherJudge)):
        return model, True

    if model in LITELLM_SUPPORTED_MODELS:
        return LiteLLMJudge(model=model), True
    if model in TOGETHER_SUPPORTED_MODELS:
        return TogetherJudge(model=model), True
    if model in JUDGMENT_SUPPORTED_MODELS:
        raise NotImplementedError(
            """Judgment models are not yet supported for local scoring.
            Please either set the `use_judgment` flag to True or use 
            non-Judgment models."""
        )
    else:
        raise InvalidJudgeModelError(f"Invalid judge model chosen: {model}")
