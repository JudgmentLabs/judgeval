from __future__ import annotations

import ast
import os
from typing import Literal, Optional, Tuple

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import UploadCustomScorerRequest
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer
from judgeval.utils.guards import expect_project_id
from judgeval.exceptions import JudgmentAPIError

RESPONSE_TYPE_MAP: dict[str, Literal["binary", "categorical", "numeric"]] = {
    "BinaryResponse": "binary",
    "CategoricalResponse": "categorical",
    "NumericResponse": "numeric",
}

V2_SCORER_BASES = {"TraceCustomScorer", "ExampleCustomScorer"}


def _extract_generic_arg(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Name):
            return node.slice.id
        if isinstance(node.slice, ast.Attribute):
            return node.slice.attr
    return None


def _get_base_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _get_base_name(node.value)
    return None


def parse_v2_scorer(
    tree: ast.AST,
) -> Optional[
    Tuple[str, Literal["trace", "example"], Literal["binary", "categorical", "numeric"]]
]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _get_base_name(base)
            if base_name not in V2_SCORER_BASES:
                continue
            generic_arg = _extract_generic_arg(base)
            if generic_arg not in RESPONSE_TYPE_MAP:
                continue
            scorer_type: Literal["trace", "example"] = (
                "trace" if base_name == "TraceCustomScorer" else "example"
            )
            return (node.name, scorer_type, RESPONSE_TYPE_MAP[generic_arg])
    return None


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

    def upload_v2(
        self,
        scorer_file_path: str,
        requirements_file_path: str | None = None,
        unique_name: str | None = None,
        overwrite: bool = False,
    ) -> bool:
        if not os.path.exists(scorer_file_path):
            raise FileNotFoundError(f"Scorer file not found: {scorer_file_path}")

        with open(scorer_file_path, "r") as f:
            scorer_code = f.read()

        try:
            tree = ast.parse(scorer_code, filename=scorer_file_path)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax in {scorer_file_path}: {e}")

        result = parse_v2_scorer(tree)
        if result is None:
            raise ValueError(
                f"No TraceCustomScorer or ExampleCustomScorer class found in {scorer_file_path}. "
                "Ensure the class inherits from TraceCustomScorer[ResponseType] or ExampleCustomScorer[ResponseType]."
            )

        class_name, scorer_type, response_type = result

        if unique_name is None:
            unique_name = class_name
            judgeval_logger.info(f"Auto-detected scorer name: '{unique_name}'")

        requirements_text = ""
        if requirements_file_path and os.path.exists(requirements_file_path):
            with open(requirements_file_path, "r") as f:
                requirements_text = f.read()

        payload = {
            "scorer_name": unique_name,
            "class_name": class_name,
            "scorer_code": scorer_code,
            "requirements_text": requirements_text,
            "overwrite": overwrite,
            "scorer_type": scorer_type,
            "response_type": response_type,
            "version": 2,
        }

        judgeval_logger.info("=" * 50)
        judgeval_logger.info("V2 Custom Scorer Upload Payload")
        judgeval_logger.info("=" * 50)
        judgeval_logger.info(f"scorer_name: {payload['scorer_name']}")
        judgeval_logger.info(f"class_name: {payload['class_name']}")
        judgeval_logger.info(f"scorer_type: {payload['scorer_type']}")
        judgeval_logger.info(f"response_type: {payload['response_type']}")
        judgeval_logger.info(f"overwrite: {payload['overwrite']}")
        judgeval_logger.info(f"version: {payload['version']}")
        judgeval_logger.info(f"requirements_text: {bool(requirements_text)}")
        judgeval_logger.info(f"scorer_code length: {len(scorer_code)} chars")
        judgeval_logger.info("=" * 50)

        return True
