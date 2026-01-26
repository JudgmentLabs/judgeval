from __future__ import annotations

from typing import Optional

from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer

import os
import ast
from judgeval.logger import judgeval_logger
from judgeval.v1.scorers.custom_scorer.utils import extract_scorer_name
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_URL


class CustomScorerFactory:
    __slots__ = ()

    def get(self, name: str, class_name: Optional[str] = None) -> CustomScorer:
        return CustomScorer(
            name=name,
            class_name=class_name or name,
            server_hosted=True,
        )

    def upload(
        self,
        scorer_file_path: str,
        requirements_file_path: str | None,
        unique_name: str | None,
        overwrite: bool = False,
        api_key: str | None = os.getenv("JUDGMENT_API_KEY"),
        organization_id: str | None = os.getenv("JUDGMENT_ORG_ID"),
    ) -> bool:
        if not os.path.exists(scorer_file_path):
            raise FileNotFoundError(f"Scorer file not found: {scorer_file_path}")

        # Auto-detect scorer name if not provided
        if unique_name is None:
            unique_name = extract_scorer_name(scorer_file_path)
            judgeval_logger.info(f"Auto-detected scorer name: '{unique_name}'")

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

        # Read requirements (optional)
        requirements_text = ""
        if requirements_file_path and os.path.exists(requirements_file_path):
            with open(requirements_file_path, "r") as f:
                requirements_text = f.read()

        try:
            if (
                not api_key
                or not api_key.strip()
                or not organization_id
                or not organization_id.strip()
            ):
                raise ValueError("Judgment API key and organization ID are required")
            client = JudgmentSyncClient(
                api_key=api_key,
                organization_id=organization_id,
                base_url=JUDGMENT_API_URL,
            )
            response = client.upload_custom_scorer(
                payload={
                    "scorer_name": unique_name,
                    "class_name": scorer_classes[0],
                    "scorer_code": scorer_code,
                    "requirements_text": requirements_text,
                    "overwrite": overwrite,
                    "scorer_type": scorer_type,
                    "version": 1,
                }
            )

            if response.get("status") == "success":
                judgeval_logger.info(
                    f"Successfully uploaded custom scorer: {unique_name}"
                )
                return True
            else:
                judgeval_logger.error(f"Failed to upload custom scorer: {unique_name}")
                return False

        except Exception:
            raise
