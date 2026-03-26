from __future__ import annotations

import ast
import io
import os
import tarfile
from typing import List, Literal, Optional, Tuple

from judgeval.cli.utils import TarFilter
from judgeval.logger import judgeval_logger
from judgeval.hosted.responses import Category
from judgeval.internal.api import JudgmentSyncClient
from judgeval.internal.api.models import (
    UploadCustomScorerBundleMetadata,
    UploadCustomScorerBundleRequest,
)

RESPONSE_TYPE_MAP: dict[str, Literal["binary", "categorical", "numeric"]] = {
    "BinaryResponse": "binary",
    "CategoricalResponse": "categorical",
    "NumericResponse": "numeric",
}

V2_SCORER_BASES = {"TraceCustomScorer", "ExampleCustomScorer"}
V3_SCORER_BASES = {"Judge"}


def _parse_category_list(node: ast.expr) -> Optional[List[Category]]:
    if not isinstance(node, ast.List):
        return None
    result = []
    for elt in node.elts:
        if not isinstance(elt, ast.Call) or elt.args:
            return None
        if not isinstance(elt.func, ast.Name) or elt.func.id != "Category":
            return None
        kw = {k.arg: k.value for k in elt.keywords}
        v = kw.get("value")
        if not isinstance(v, ast.Constant) or not isinstance(v.value, str):
            return None
        d = kw.get("description")
        desc = (
            d.value if isinstance(d, ast.Constant) and isinstance(d.value, str) else ""
        )
        result.append(Category(value=v.value, description=desc))
    return result or None


def _extract_generic_arg(
    node: ast.expr,
    tree: ast.AST,
) -> Tuple[Optional[str], Optional[List[Category]]]:
    name = None
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Name):
            name = node.slice.id
        elif isinstance(node.slice, ast.Attribute):
            name = node.slice.attr
    if name is None:
        return (None, None)

    if name in RESPONSE_TYPE_MAP:
        if name == "CategoricalResponse":
            raise ValueError(
                "Judge[CategoricalResponse] is not supported. "
                "Define a CategoricalResponse subclass with categories."
            )
        return (name, None)

    for resolved in ast.walk(tree):
        if not isinstance(resolved, ast.ClassDef) or resolved.name != name:
            continue
        for base in resolved.bases:
            base_name = _get_base_name(base)
            if base_name not in RESPONSE_TYPE_MAP:
                continue
            if base_name != "CategoricalResponse":
                return base_name, None
            for item in resolved.body:
                if not isinstance(item, ast.Assign):
                    continue
                for target in item.targets:
                    if not (isinstance(target, ast.Name) and target.id == "categories"):
                        continue
                    categories = _parse_category_list(item.value)
                    if categories is not None:
                        return (base_name, categories)
                    return (None, None)
            return (base_name, None)
    return (None, None)


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
        Optional[List[Category]],
    ]
]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _get_base_name(base)
            if base_name not in V2_SCORER_BASES and base_name not in V3_SCORER_BASES:
                continue
            generic_arg, categories = _extract_generic_arg(base, tree)
            if generic_arg not in RESPONSE_TYPE_MAP:
                continue
            if base_name in V3_SCORER_BASES:
                return (node.name, None, RESPONSE_TYPE_MAP[generic_arg], categories)
            scorer_type: Literal["trace", "example"] = (
                "trace" if base_name == "TraceCustomScorer" else "example"
            )
            return (node.name, scorer_type, RESPONSE_TYPE_MAP[generic_arg], categories)
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


def upload_judge(
    client: JudgmentSyncClient,
    project_id: str,
    entrypoint_path: str,
    included_files_paths: list[str] | None = None,
    requirements_file_path: str | None = None,
    unique_name: str | None = None,
    bump_major: bool = False,
    project_name: str | None = None,
    yes: bool = False,
) -> bool:
    if included_files_paths is None:
        included_files_paths = []

    if not os.path.exists(entrypoint_path):
        raise FileNotFoundError(f"Scorer file not found: {entrypoint_path}")

    with open(entrypoint_path, "r") as f:
        scorer_code = f.read()

    try:
        tree = ast.parse(scorer_code, filename=entrypoint_path)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax in {entrypoint_path}: {e}")

    result = parse_judge(tree)
    if result is None:
        raise ValueError(
            f"No valid Judge, TraceCustomScorer, or ExampleCustomScorer class found in {entrypoint_path}. "
            "Ensure the class inherits from Judge[ResponseType], TraceCustomScorer[ResponseType], "
            "or ExampleCustomScorer[ResponseType].\n\n"
            "For categorical response types, define a CategoricalResponse subclass with a 'categories' "
            "class variable as a list of Category models, then use it as the generic argument. For example:\n\n"
            "class MyResponse(CategoricalResponse):\n"
            "    categories = [\n"
            "        Category(value='Passed', description='The agent passed the test'),\n"
            "        Category(value='Not Passed', description='The agent failed the test'),\n"
            "    ]\n\n"
            "class CategoricalScorer(Judge[MyResponse]):\n"
            "    async def score(self, data: Example) -> MyResponse:\n "
            "        return MyResponse(value='Passed', reason='The agent passed the test')\n"
        )

    class_name, scorer_type, response_type, categories = result

    if response_type == "categorical" and categories is None:
        raise ValueError(
            f"Categorical response type requires categories to be defined in {entrypoint_path}. "
            "Ensure the class defines a 'categories' class variable as a list of Category models."
        )

    if unique_name is None:
        unique_name = class_name
        judgeval_logger.info(f"Auto-detected judge name: '{unique_name}'")

    bundle, entrypoint_arcname, requirements_arcname, file_count = _build_bundle(
        entrypoint_path, included_files_paths, requirements_file_path
    )

    if not yes:
        import typer

        display_name = f"'{project_name}'" if project_name else project_id
        typer.confirm(
            f"Are you sure you want to upload {response_type} code judge '{unique_name}' to project {display_name}? "
            f"In total, {file_count} files will be uploaded.\n"
            f"If this judge already exists in the project, a new version will be created.",
            abort=True,
        )

    metadata: UploadCustomScorerBundleMetadata = {
        "scorer_name": unique_name,
        "entrypoint_path": entrypoint_arcname,
        "class_name": class_name,
        "scorer_type": scorer_type,
        "response_type": response_type,
        "version": 3 if scorer_type is None else 2,
        "categories": [category.model_dump() for category in categories]
        if categories
        else None,
        "bump_major": bump_major,
    }
    if requirements_arcname:
        metadata["requirements_path"] = requirements_arcname

    payload: UploadCustomScorerBundleRequest = {
        "metadata": metadata,
        "bundle": bundle,
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
