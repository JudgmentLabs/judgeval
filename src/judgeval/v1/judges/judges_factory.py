from __future__ import annotations

import ast
import os
from typing import Dict, Literal, Optional, Tuple

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.models import (
    FetchPromptScorersResponse,
    PromptScorer as APIPromptScorer,
    UploadCustomScorerRequest,
)
from judgeval.exceptions import JudgmentAPIError
from judgeval.v1.judges.base_judge import BaseJudge
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id

RESPONSE_TYPE_MAP: dict[str, Literal["binary", "categorical", "numeric"]] = {
    "BinaryResponse": "binary",
    "CategoricalResponse": "categorical",
    "NumericResponse": "numeric",
}

_JUDGE_BASES = {"Judge", "TraceCustomScorer", "ExampleCustomScorer"}


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


def parse_judge(
    tree: ast.AST,
) -> Optional[
    Tuple[
        str,
        Optional[Literal["trace", "example"]],
        Literal["binary", "categorical", "numeric"],
    ]
]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _get_base_name(base)
            if base_name not in _JUDGE_BASES:
                continue
            generic_arg = _extract_generic_arg(base)
            if generic_arg not in RESPONSE_TYPE_MAP:
                continue
            if base_name == "Judge":
                return (node.name, None, RESPONSE_TYPE_MAP[generic_arg])
            scorer_type: Literal["trace", "example"] = (
                "trace" if base_name == "TraceCustomScorer" else "example"
            )
            return (node.name, scorer_type, RESPONSE_TYPE_MAP[generic_arg])
    return None


class JudgesFactory:
    __slots__ = ("_client", "_project_id")
    _cache: Dict[Tuple[str, str, str, str], APIPromptScorer] = {}

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
    ):
        self._client = client
        self._project_id = project_id

    def get(self, name: str) -> BaseJudge | None:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        cache_key = (
            name,
            self._client.organization_id,
            self._client.api_key,
            project_id,
        )
        cached = self._cache.get(cache_key)

        if cached is None:
            try:
                response: FetchPromptScorersResponse = (
                    self._client.get_projects_scorers(
                        project_id=project_id,
                        names=name,
                        is_trace="true",
                    )
                )
                scorers = response.get("scorers", [])

                if not scorers:
                    raise JudgmentAPIError(
                        404, f"Failed to fetch judge '{name}': not found", None
                    )

                scorer = scorers[0]
                self._cache[cache_key] = scorer
                cached = scorer
            except JudgmentAPIError:
                judgeval_logger.error(
                    f"Failed to fetch judge '{name}': judge '{name}' not found in the organization."
                )
                return None
            except Exception:
                judgeval_logger.error(f"Failed to fetch judge '{name}'.")
                return None

        return BaseJudge(
            name=name,
            prompt=cached.get("prompt", ""),
            threshold=cached.get("threshold", 0.5),
            options=cached.get("options"),
            model=cached.get("model"),
            description=cached.get("description"),
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

        with open(scorer_file_path, "r") as f:
            scorer_code = f.read()

        try:
            tree = ast.parse(scorer_code, filename=scorer_file_path)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax in {scorer_file_path}: {e}")

        result = parse_judge(tree)
        if result is None:
            raise ValueError(
                f"No Judge class found in {scorer_file_path}. "
                "Ensure the class inherits from Judge[ResponseType]."
            )

        class_name, scorer_type, response_type = result

        if unique_name is None:
            unique_name = class_name
            judgeval_logger.info(f"Auto-detected judge name: '{unique_name}'")

        requirements_text = ""
        if requirements_file_path and os.path.exists(requirements_file_path):
            with open(requirements_file_path, "r") as f:
                requirements_text = f.read()

        if not overwrite:
            try:
                exists_resp = self._client.get_projects_scorers_custom_by_name_exists(
                    project_id=project_id, name=unique_name
                )
                if exists_resp.get("exists"):
                    raise JudgmentAPIError(
                        status_code=409,
                        detail=f"Judge '{unique_name}' already exists. Use --overwrite to replace.",
                        response=None,
                    )
            except JudgmentAPIError as e:
                if e.status_code == 409:
                    raise

        payload: UploadCustomScorerRequest = {
            "scorer_name": unique_name,
            "class_name": class_name,
            "scorer_code": scorer_code,
            "requirements_text": requirements_text,
            "overwrite": overwrite,
            "scorer_type": scorer_type,
            "response_type": response_type,
            "version": 3 if scorer_type is None else 2,
        }
        response = self._client.post_projects_scorers_custom(
            project_id=project_id,
            payload=payload,
        )

        if response.get("status") == "success":
            judgeval_logger.info(f"Successfully uploaded custom judge: {unique_name}")
            return True
        else:
            judgeval_logger.error(f"Failed to upload custom judge: {unique_name}")
            return False
