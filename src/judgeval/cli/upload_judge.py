from __future__ import annotations

import ast
import io
import os
import tarfile
from typing import Literal, Optional, Tuple

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.models import (
    UploadCustomScorerBundleMetadata,
    UploadCustomScorerBundleRequest,
)

RESPONSE_TYPE_MAP: dict[str, Literal["binary", "categorical", "numeric"]] = {
    "BinaryResponse": "binary",
    "CategoricalResponse": "categorical",
    "NumericResponse": "numeric",
}


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
) -> Optional[Tuple[str, Literal["binary", "categorical", "numeric"]]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _get_base_name(base)
            if base_name != "Judge":
                continue
            generic_arg = _extract_generic_arg(base)
            if generic_arg not in RESPONSE_TYPE_MAP:
                continue
            return (node.name, RESPONSE_TYPE_MAP[generic_arg])
    return None


def upload_judge(
    client: JudgmentSyncClient,
    project_id: str,
    scorer_file_path: str,
    requirements_file_path: str | None = None,
    unique_name: str | None = None,
    bump_major: bool = False,
) -> bool:
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

    class_name, response_type = result

    if unique_name is None:
        unique_name = class_name
        judgeval_logger.info(f"Auto-detected judge name: '{unique_name}'")

    entrypoint_arcname = os.path.basename(scorer_file_path)
    requirements_arcname = (
        os.path.basename(requirements_file_path) if requirements_file_path else None
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.GNU_FORMAT) as tar:
        tar.add(scorer_file_path, arcname=entrypoint_arcname)
        if requirements_file_path:
            if not os.path.exists(requirements_file_path):
                raise FileNotFoundError(
                    f"Requirements file not found: {requirements_file_path}"
                )
            tar.add(requirements_file_path, arcname=requirements_arcname)

    metadata: UploadCustomScorerBundleMetadata = {
        "scorer_name": unique_name,
        "entrypoint_path": entrypoint_arcname,
        "class_name": class_name,
        "response_type": response_type,
        "version": 3,
        "bump_major": bump_major,
    }
    if requirements_arcname:
        metadata["requirements_path"] = requirements_arcname

    payload: UploadCustomScorerBundleRequest = {
        "metadata": metadata,
        "bundle": buf.getvalue(),
    }

    response = client.post_projects_scorers_custom_bundle(
        project_id=project_id,
        payload=payload,
    )

    if response.get("status") == "success":
        judgeval_logger.info(f"Successfully uploaded custom judge: {unique_name}")
        return True
    else:
        judgeval_logger.error(f"Failed to upload custom judge: {unique_name}")
        return False
