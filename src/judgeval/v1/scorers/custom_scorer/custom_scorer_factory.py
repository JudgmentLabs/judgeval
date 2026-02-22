from __future__ import annotations

import ast
import io
import os
import tarfile
from typing import Literal, Optional, Tuple

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import (
    UploadCustomScorerMetadata,
    UploadCustomScorerRequest,
)
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer
from judgeval.utils.guards import expect_project_id
from judgeval.exceptions import JudgmentAPIError
import typer

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


def _build_bundle(
    entrypoint_path: str,
    included_files_paths: list[str],
    requirements_file_path: str | None,
) -> bytes:
    all_abs: list[str] = [os.path.abspath(entrypoint_path)]
    for p in included_files_paths:
        all_abs.append(os.path.abspath(p))
    if requirements_file_path and os.path.exists(requirements_file_path):
        all_abs.append(os.path.abspath(requirements_file_path))

    base_dirs = [os.path.dirname(p) if os.path.isfile(p) else p for p in all_abs]
    common = os.path.commonpath(base_dirs)

    seen: set[str] = set()

    def dedup_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if tarinfo.name in seen:
            return None
        seen.add(tarinfo.name)
        return tarinfo

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.GNU_FORMAT) as tar:
        for abs_path in all_abs:
            arcname = os.path.relpath(abs_path, common)
            tar.add(abs_path, arcname=arcname, filter=dedup_filter)

    return buf.getvalue()


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
        entrypoint_path: str,
        included_files_paths: list[str],
        requirements_file_path: str | None = None,
        unique_name: str | None = None,
    ) -> bool:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return False

        if not os.path.exists(entrypoint_path):
            raise FileNotFoundError(
                f"Scorer entrypoint file not found: {entrypoint_path}"
            )

        with open(entrypoint_path, "r") as f:
            scorer_code = f.read()

        try:
            tree = ast.parse(scorer_code, filename=entrypoint_path)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax in {entrypoint_path}: {e}")

        result = parse_v2_scorer(tree)
        if result is None:
            raise ValueError(
                f"No TraceCustomScorer or ExampleCustomScorer class found in {entrypoint_path}. "
                "Ensure the class inherits from TraceCustomScorer[ResponseType] or ExampleCustomScorer[ResponseType]."
            )

        py_files: list[str] = []
        for path in included_files_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Included path not found: {path}")
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for fname in files:
                        if fname.endswith(".py"):
                            py_files.append(os.path.join(root, fname))
            elif path.endswith(".py"):
                py_files.append(path)

        for file_path in py_files:
            with open(file_path, "r") as f:
                code = f.read()
            try:
                ast.parse(code, filename=file_path)
            except SyntaxError as e:
                raise ValueError(f"Invalid Python syntax in {file_path}: {e}")

        class_name, scorer_type, response_type = result

        if unique_name is None:
            unique_name = class_name
            judgeval_logger.info(f"Auto-detected scorer name: '{unique_name}'")

        exists_resp = self._client.get_projects_scorers_custom_by_name_exists(
            project_id=project_id, name=unique_name
        )
        if exists_resp.get("exists"):
            typer.confirm(
                f"Scorer '{unique_name}' already exists. Upload a new version?",
                abort=True,
            )

        bundle = _build_bundle(
            entrypoint_path, included_files_paths, requirements_file_path
        )

        all_abs = [os.path.abspath(entrypoint_path)]
        all_abs.extend(os.path.abspath(p) for p in included_files_paths)
        base_dirs = [os.path.dirname(p) if os.path.isfile(p) else p for p in all_abs]
        common = os.path.commonpath(base_dirs)
        entrypoint_arcname = os.path.relpath(os.path.abspath(entrypoint_path), common)
        requirements_arcname = (
            os.path.relpath(os.path.abspath(requirements_file_path), common)
            if requirements_file_path
            else None
        )

        metadata: UploadCustomScorerMetadata = {
            "scorer_name": unique_name,
            "entrypoint_path": entrypoint_arcname,
            "class_name": class_name,
            "scorer_type": scorer_type,
            "response_type": response_type,
            "version": 2,
        }

        if requirements_arcname:
            metadata["requirements_path"] = requirements_arcname

        payload: UploadCustomScorerRequest = {
            "metadata": metadata,
            "bundle": bundle,
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
