from __future__ import annotations

import ast
import io
import os
import tarfile
from pathlib import Path
from typing import Literal, Optional, Tuple
from pathspec import PathSpec

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import (
    UploadCustomScorerBundleMetadata,
    UploadCustomScorerBundleRequest,
)
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer
from judgeval.utils.guards import expect_project_id
from judgeval.exceptions import JudgmentAPIError

RESPONSE_TYPE_MAP: dict[str, Literal["binary", "categorical", "numeric"]] = {
    "BinaryResponse": "binary",
    "CategoricalResponse": "categorical",
    "NumericResponse": "numeric",
}

V2_SCORER_BASES = {"TraceCustomScorer", "ExampleCustomScorer"}
V3_SCORER_BASES = {"Judge"}

EXCLUDE_SPEC = PathSpec.from_lines(
    "gitignore",
    [
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "**/*.pyw",
        "*.pyz",
        ".venv/",
        "venv/",
        ".env",
        ".env.*",
    ],
)


def _find_gitignore_path(start_path: str) -> str | None:
    """Walk up from start_path to find directory containing .gitignore."""
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent
    while current != current.parent:
        if (current / ".gitignore").is_file():
            return str(current / ".gitignore")
        current = current.parent
    return None


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
) -> tuple[bytes, str, str | None]:
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

    base_dirs = [os.path.dirname(p) if os.path.isfile(p) else p for p in all_abs]
    common = os.path.commonpath(base_dirs)
    gitignore_path = _find_gitignore_path(common)
    gitignore_spec = None
    if gitignore_path:
        with open(gitignore_path, "r") as f:
            gitignore_spec = PathSpec.from_lines("gitignore", f)

    def should_exclude(path: str) -> bool:
        exclude_pattern_matches = EXCLUDE_SPEC.match_file(path)
        if gitignore_spec and gitignore_path:
            abs_path = os.path.join(common, path)
            rel_to_gitignore = os.path.relpath(
                abs_path, os.path.dirname(gitignore_path)
            )
            gitignore_matches = gitignore_spec.match_file(rel_to_gitignore)
        else:
            gitignore_matches = False
        return exclude_pattern_matches or gitignore_matches

    seen: set[str] = set()

    def dedup_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if tarinfo.name in seen or should_exclude(tarinfo.name):
            return None
        seen.add(tarinfo.name)
        if tarinfo.isfile() and tarinfo.name.endswith(".py"):
            full_path = os.path.join(common, tarinfo.name)
            with open(full_path, "r") as f:
                try:
                    ast.parse(f.read(), filename=full_path)
                except SyntaxError as e:
                    raise ValueError(
                        f"Invalid Python syntax in {full_path}: {e}"
                    ) from e
        return tarinfo

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.GNU_FORMAT) as tar:
        for abs_path in all_abs:
            arcname = os.path.relpath(abs_path, common)
            tar.add(abs_path, arcname=arcname, filter=dedup_filter)

    entrypoint_arcname = os.path.relpath(os.path.abspath(entrypoint_path), common)
    requirements_arcname = (
        os.path.relpath(os.path.abspath(requirements_file_path), common)
        if requirements_file_path
        else None
    )

    return buf.getvalue(), entrypoint_arcname, requirements_arcname


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
        bump_major: bool = False,
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

        bundle, entrypoint_arcname, requirements_arcname = _build_bundle(
            entrypoint_path, included_files_paths, requirements_file_path
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
