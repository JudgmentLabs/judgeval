from __future__ import annotations

import ast
import os
from typing import Optional

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import UploadCustomScorerRequest
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer
from judgeval.utils.guards import expect_project_id
from judgeval.exceptions import JudgmentAPIError


class CustomScorerFactory:
    __slots__ = ("_client", "_project_id")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
    ):
        self._client = client
        self._project_id = project_id

    def get(self, name: str) -> Optional[CustomScorer]:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        scorer_exists = self._client.get_projects_scorers_custom_by_name_exists(
            project_id=project_id, name=name
        )
        if not scorer_exists["exists"]:
            raise JudgmentAPIError(
                status_code=404, detail=f"Scorer {name} does not exist", response=None
            )

        return CustomScorer(
            name=name,
            project_id=project_id,
        )

    def upload(
        self,
        scorer_file_path: str,
        requirements_file_path: str | None = None,
        unique_name: str | None = None,
        overwrite: bool = False,
    ) -> bool:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return False

        if not os.path.exists(scorer_file_path):
            raise FileNotFoundError(f"Scorer file not found: {scorer_file_path}")

        # Read scorer code
        with open(scorer_file_path, "r") as f:
            scorer_code = f.read()

        try:
            tree = ast.parse(scorer_code, filename=scorer_file_path)
        except SyntaxError as e:
            error_msg = f"Invalid Python syntax in {scorer_file_path}: {e}"
            judgeval_logger.error(error_msg)
            raise ValueError(error_msg)

        scorer_classes = []
        scorer_type = "example"
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if (isinstance(base, ast.Name) and base.id == "ExampleScorer") or (
                        isinstance(base, ast.Attribute) and base.attr == "ExampleScorer"
                    ):
                        scorer_classes.append(node.name)
                    if (isinstance(base, ast.Name) and base.id == "TraceScorer") or (
                        isinstance(base, ast.Attribute) and base.attr == "TraceScorer"
                    ):
                        scorer_classes.append(node.name)
                        scorer_type = "trace"

        if len(scorer_classes) > 1:
            error_msg = f"Multiple ExampleScorer/TraceScorer classes found in {scorer_file_path}: {scorer_classes}. Please only upload one scorer class per file."
            judgeval_logger.error(error_msg)
            raise ValueError(error_msg)
        elif len(scorer_classes) == 0:
            error_msg = f"No ExampleScorer or TraceScorer class was found in {scorer_file_path}. Please ensure the file contains a valid scorer class that inherits from ExampleScorer or TraceScorer."
            judgeval_logger.error(error_msg)
            raise ValueError(error_msg)

        # Auto-detect scorer name if not provided
        if unique_name is None:
            unique_name = scorer_classes[0]
            judgeval_logger.info(f"Auto-detected scorer name: '{unique_name}'")

        # Read requirements (optional)
        requirements_text = ""
        if requirements_file_path and os.path.exists(requirements_file_path):
            with open(requirements_file_path, "r") as f:
                requirements_text = f.read()

        payload: UploadCustomScorerRequest = {
            "scorer_name": unique_name,
            "class_name": scorer_classes[0],
            "scorer_code": scorer_code,
            "requirements_text": requirements_text,
            "overwrite": overwrite,
            "scorer_type": scorer_type,
            "version": 1,
        }
        response = self._client.post_projects_scorers_custom(
            project_id=project_id,
            payload=payload,
        )

        if response.get("status") == "success":
            judgeval_logger.info(f"Successfully uploaded custom scorer: {unique_name}")
            return True
        else:
            judgeval_logger.error(f"Failed to upload custom scorer: {unique_name}")
            return False
