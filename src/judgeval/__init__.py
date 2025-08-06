from __future__ import annotations

from judgeval.evaluation import EvaluationRun


from typing import List, Never, Optional
from judgeval.data.judgment_types import ExampleJudgmentType, ScoringResultJudgmentType
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_DEFAULT_GPT_MODEL, JUDGMENT_ORG_ID
from judgeval.utils.meta import SingletonMeta
from judgeval.exceptions import JudgmentRuntimeError


class JudgmentClient(metaclass=SingletonMeta):
    __slots__ = ("api_key", "organization_id")

    def __init__(
        self,
        api_key: Optional[str],
        organization_id: Optional[str],
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
        examples: List[ExampleJudgmentType],
        scorers: List[Never],
        project_name: str,
        eval_run_name: str,
        model: str = JUDGMENT_DEFAULT_GPT_MODEL,
        override: bool = False,
        append: bool = False,
        async_execute: bool = False,
    ) -> List[ScoringResultJudgmentType]:
        ...

        if override and append:
            raise ValueError("Cannot override and append at the same time")

        try:
            eval = EvaluationRun(
                append=append,
                override=override,
                project_name=project_name,
                eval_name=eval_run_name,
                examples=examples,
                scorers=scorers,
                model=model,
                organization_id=self.organization_id,
            )

            return run_evaluation(eval)

        except ValueError as e:
            raise ValueError(
                f"Please check your EvaluationRun object, one or more fields are invalid: \n{e}"
            )

        except Exception as e:
            raise JudgmentRuntimeError(
                f"An unexpected error occured during evaluation: {e}"
            ) from e


__all__ = ("JudgmentClient",)
