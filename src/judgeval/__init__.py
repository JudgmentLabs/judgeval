from __future__ import annotations

from judgeval.data.result import ScoringResult
from judgeval.evaluation import run_eval
from judgeval.data.evaluation_run import EvaluationRun


from typing import List, Optional, Union
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.data.example import Example
from judgeval.logger import judgeval_logger
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_DEFAULT_GPT_MODEL, JUDGMENT_ORG_ID
from judgeval.utils.meta import SingletonMeta
from judgeval.exceptions import JudgmentRuntimeError
from judgeval.utils.file_utils import extract_scorer_name
from judgeval.api import JudgmentSyncClient


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

    def upload_custom_scorer(
        self,
        scorer_file_path: str,
        requirements_file_path: Optional[str] = None,
        unique_name: Optional[str] = None,
    ) -> bool:
        """
        Upload custom ExampleScorer from files to backend.

        Args:
            scorer_file_path: Path to Python file containing CustomScorer class
            requirements_file_path: Optional path to requirements.txt
            unique_name: Optional unique identifier (auto-detected from scorer.name if not provided)

        Returns:
            bool: True if upload successful

        Raises:
            ValueError: If scorer file is invalid
            FileNotFoundError: If scorer file doesn't exist
        """
        import os

        if not os.path.exists(scorer_file_path):
            raise FileNotFoundError(f"Scorer file not found: {scorer_file_path}")

        # Auto-detect scorer name if not provided
        if unique_name is None:
            unique_name = extract_scorer_name(scorer_file_path)
            judgeval_logger.info(f"Auto-detected scorer name: '{unique_name}'")

        # Read scorer code
        with open(scorer_file_path, "r") as f:
            scorer_code = f.read()

        # Read requirements (optional)
        requirements_text = ""
        if requirements_file_path and os.path.exists(requirements_file_path):
            with open(requirements_file_path, "r") as f:
                requirements_text = f.read()

        try:
            client = JudgmentSyncClient(
                api_key=JUDGMENT_API_KEY, organization_id=JUDGMENT_ORG_ID
            )
            response = client.upload_custom_scorer(
                scorer_name=unique_name,
                scorer_code=scorer_code,
                requirements_text=requirements_text,
            )

            if response.get("status") == "success":
                judgeval_logger.info(
                    f"Successfully uploaded custom scorer: {unique_name}"
                )
                return True
            else:
                judgeval_logger.error(f"Failed to upload custom scorer: {unique_name}")
                return False

        except Exception as e:
            judgeval_logger.error(f"Error uploading custom scorer: {e}")
            raise


__all__ = ("JudgmentClient",)
