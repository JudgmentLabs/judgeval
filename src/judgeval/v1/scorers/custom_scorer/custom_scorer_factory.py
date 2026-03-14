from __future__ import annotations

import ast
import io
import os
import tarfile
from typing import Literal, Optional, Tuple
import typer

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import (
    UploadCustomScorerBundleMetadata,
    UploadCustomScorerBundleRequest,
)
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer
from judgeval.utils.guards import expect_project_id
from judgeval.exceptions import JudgmentAPIError
from judgeval.v1.scorers.custom_scorer.utils import TarFilter

RESPONSE_TYPE_MAP: dict[str, Literal["binary", "categorical", "numeric"]] = {
    "BinaryResponse": "binary",
    "CategoricalResponse": "categorical",
    "NumericResponse": "numeric",
}

V2_SCORER_BASES = {"TraceCustomScorer", "ExampleCustomScorer"}
V3_SCORER_BASES = {"Judge"}


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
            if base_name not in V2_SCORER_BASES and base_name not in V3_SCORER_BASES:
                continue
            generic_arg = _extract_generic_arg(base)
            if generic_arg not in RESPONSE_TYPE_MAP:
                continue
            if base_name in V3_SCORER_BASES:
                return (node.name, None, RESPONSE_TYPE_MAP[generic_arg])
            scorer_type: Literal["trace", "example"] = (
                "trace" if base_name == "TraceCustomScorer" else "example"
            )
            return (node.name, scorer_type, RESPONSE_TYPE_MAP[generic_arg])
    return None


def _build_bundle(
    entrypoint_path: str,
    included_files_paths: list[str],
    requirements_file_path: str | None,
) -> tuple[bytes, str, str | None, int]:
    if not os.path.exists(entrypoint_path):
        raise FileNotFoundError(f"Scorer entrypoint file not found: {entrypoint_path}")
    all_abs: list[str] = [os.path.abspath(entrypoint_path)]

    for p in included_files_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Included path not found: {p}")
        all_abs.append(os.path.abspath(p))

    if requirements_file_path:
        if not os.path.exists(requirements_file_path):
            raise FileNotFoundError(
                f"Specified requirements file not found: {requirements_file_path}"
            )
        all_abs.append(os.path.abspath(requirements_file_path))

    base_dirs = [os.path.dirname(p) if os.path.isfile(p) else p for p in all_abs] + [
        os.path.abspath(os.path.curdir)
    ]
    common_base_dir = os.path.commonpath(base_dirs)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.GNU_FORMAT) as tar:
        tar_filter = TarFilter(common_base_dir)
        for abs_path in all_abs:
            arcname = os.path.relpath(abs_path, common_base_dir)
            tar.add(abs_path, arcname=arcname, filter=tar_filter)

    entrypoint_arcname = os.path.relpath(
        os.path.abspath(entrypoint_path), common_base_dir
    )
    requirements_arcname = (
        os.path.relpath(os.path.abspath(requirements_file_path), common_base_dir)
        if requirements_file_path
        else None
    )

    return (
        buf.getvalue(),
        entrypoint_arcname,
        requirements_arcname,
        tar_filter.get_file_count(),
    )


class CustomScorerFactory:
    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

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
        bump_major: bool = False,
        yes: bool = False,
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

        result = parse_judge(tree)
        if result is None:
            raise ValueError(
                f"No Judge, TraceCustomScorer, or ExampleCustomScorer class found in {entrypoint_path}. "
                "Ensure the class inherits from Judge[ResponseType], TraceCustomScorer[ResponseType], "
                "or ExampleCustomScorer[ResponseType]."
            )

        class_name, scorer_type, response_type = result

        if unique_name is None:
            unique_name = class_name
            judgeval_logger.info(f"Auto-detected scorer name: '{unique_name}'")

        bundle, entrypoint_arcname, requirements_arcname, file_count = _build_bundle(
            entrypoint_path, included_files_paths, requirements_file_path
        )

        if not yes:
            typer.confirm(
                f"Are you sure you want to upload {response_type} code judge '{unique_name}' to project '{self._project_name}'? In total, {file_count} files will be uploaded.\nIf this judge already exists in the project, a new version will be created.",
                abort=True,
            )

        metadata: UploadCustomScorerBundleMetadata = {
            "scorer_name": unique_name,
            "entrypoint_path": entrypoint_arcname,
            "class_name": class_name,
            "scorer_type": scorer_type,
            "response_type": response_type,
            "version": 3 if scorer_type is None else 2,
            "bump_major": bump_major,
        }

        if requirements_arcname:
            metadata["requirements_path"] = requirements_arcname

        payload: UploadCustomScorerBundleRequest = {
            "metadata": metadata,
            "bundle": bundle,
        }

        response = self._client.post_projects_scorers_custom_bundle(
            project_id=project_id,
            payload=payload,
        )

        if response.get("status") == "success":
            judgeval_logger.info(f"Successfully uploaded custom scorer: {unique_name}")
            return True
        else:
            judgeval_logger.error(f"Failed to upload custom scorer: {unique_name}")
            return False
