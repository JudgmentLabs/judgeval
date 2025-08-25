"""
Util functions for Scorer objects
"""

from typing import List

from judgeval.scorers import BaseScorer
from judgeval.data import Example, ExampleParams
from judgeval.scorers.exceptions import MissingExampleParamsError


def clone_scorers(scorers: List[BaseScorer]) -> List[BaseScorer]:
    """
    Creates duplicates of the scorers passed as argument.
    """
    cloned_scorers = []
    for s in scorers:
        cloned_scorers.append(s.model_copy(deep=True))
    return cloned_scorers


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
