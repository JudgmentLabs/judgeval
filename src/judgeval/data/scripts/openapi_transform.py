import json
import sys
from typing import Any, Dict, Generator, List
import requests

spec_file = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/openapi.json"

if spec_file.startswith("http"):
    r = requests.get(spec_file)
    r.raise_for_status()
    SPEC = r.json()
else:
    with open(spec_file, "r") as f:
        SPEC = json.load(f)

JUDGEVAL_PATHS: List[str] = [
    "/log_eval_results/",
]


def resolve_ref(ref: str) -> str:
    assert ref.startswith("#/components/schemas/"), (
        "Reference must start with #/components/schemas/"
    )
    return ref.replace("#/components/schemas/", "")


def walk(obj: Any) -> Generator[Any, None, None]:
    yield obj
    if isinstance(obj, list):
        for item in obj:
            yield from walk(item)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from walk(value)


def get_referenced_schemas(obj: Any) -> Generator[str, None, None]:
    for value in walk(obj):
        if isinstance(value, dict) and "$ref" in value:
            ref = value["$ref"]
            resolved = resolve_ref(ref)
            assert isinstance(ref, str), "Reference must be a string"
            # Strip the _JudgmentType suffix if it exists to get the original schema name
            if resolved.endswith("_JudgmentType"):
                resolved = resolved[: -len("_JudgmentType")]
            yield resolved


def transform_schema_refs(obj: Any) -> Any:
    """Transform all $ref values in a schema to use the _JudgmentType suffix"""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if (
                key == "$ref"
                and isinstance(value, str)
                and value.startswith("#/components/schemas/")
            ):
                # Update the reference to use the suffixed name
                original_name = resolve_ref(value)
                suffixed_name = f"{original_name}_JudgmentType"
                result[key] = f"#/components/schemas/{suffixed_name}"
            else:
                result[key] = transform_schema_refs(value)
        return result
    elif isinstance(obj, list):
        return [transform_schema_refs(item) for item in obj]
    else:
        return obj


filtered_paths = {
    path: spec_data
    for path, spec_data in SPEC["paths"].items()
    if path in JUDGEVAL_PATHS
}


def filter_schemas() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    processed_original_names: set[str] = set()
    schemas_to_scan: Any = filtered_paths

    while True:
        to_commit: Dict[str, Any] = {}
        for original_schema_name in get_referenced_schemas(schemas_to_scan):
            if original_schema_name in processed_original_names:
                continue

            assert original_schema_name in SPEC["components"]["schemas"], (
                f"Schema {original_schema_name} not found in components.schemas"
            )
            # Transform the schema to update any internal references
            original_schema = SPEC["components"]["schemas"][original_schema_name]
            transformed_schema = transform_schema_refs(original_schema)
            suffixed_name = f"{original_schema_name}_JudgmentType"
            to_commit[suffixed_name] = transformed_schema
            processed_original_names.add(original_schema_name)

        if not to_commit:
            break

        result.update(to_commit)
        schemas_to_scan = to_commit

    return result


# Transform the filtered paths to update schema references
transformed_paths = transform_schema_refs(filtered_paths)

spec = {
    "openapi": SPEC["openapi"],
    "info": SPEC["info"],
    "paths": transformed_paths,
    "components": {
        **SPEC["components"],
        "schemas": filter_schemas(),
    },
}

print(json.dumps(spec, indent=4))
