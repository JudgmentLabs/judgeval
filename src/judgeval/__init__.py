from __future__ import annotations

from judgeval.data.result import ScoringResult
from judgeval.evaluation import run_eval
from judgeval.data.evaluation_run import EvaluationRun


from typing import List, Never, Optional, Union
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.data.example import Example
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_DEFAULT_GPT_MODEL, JUDGMENT_ORG_ID
from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.utils.meta import SingletonMeta
from judgeval.exceptions import JudgmentRuntimeError


class JudgmentClient(metaclass=SingletonMeta):
    __slots__ = ("api_key", "organization_id")

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
    ):
        _api_key = api_key or JUDGMENT_API_KEY
        _organization_id = organization_id or JUDGMENT_ORG_ID

        if _api_key is None:
            raise ValueError(
                "API Key is not set, please set JUDGMENT_API_KEY in the environment variables or pass it as `api_key` "
            )

        if _organization_id is None:
            raise ValueError(
                "Organization ID is not set, please set JUDGMENT_ORG_ID in the environment variables or pass it as `organization_id`"
            )

        self.api_key = _api_key
        self.organization_id = _organization_id

    def run_evaluation(
        self,
        examples: List[Example],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        project_name: str,
        eval_run_name: str,
        model: str = JUDGMENT_DEFAULT_GPT_MODEL,
    ) -> List[ScoringResult]:
        ...

        try:
            eval = EvaluationRun(
                project_name=project_name,
                eval_name=eval_run_name,
                examples=examples,
                scorers=scorers,
                model=model,
                organization_id=self.organization_id,
            )

            return run_eval(eval, self.api_key)

        except ValueError as e:
            raise ValueError(
                f"Please check your EvaluationRun object, one or more fields are invalid: \n{e}"
            )

        except Exception as e:
            raise JudgmentRuntimeError(
                f"An unexpected error occured during evaluation: {e}"
            ) from e


__all__ = ("JudgmentClient",)
