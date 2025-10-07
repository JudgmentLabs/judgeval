from __future__ import annotations

import orjson
import sys
from typing import Any, Dict, Generator, List
import httpx

spec_file = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/openapi.json"

if spec_file.startswith("http"):
    r = httpx.get(spec_file)
    r.raise_for_status()
    SPEC = r.json()
else:
    with open(spec_file, "rb") as f:
        SPEC = orjson.loads(f.read())

JUDGEVAL_PATHS: List[str] = [
    "/traces/spans/batch/",
    "/traces/evaluation_runs/batch/",
    "/traces/fetch/",
    "/traces/upsert/",
    "/traces/add_to_dataset/",
    "/projects/add/",
    "/projects/delete_from_judgeval/",
    "/evaluate/traces",
    "/evaluate/examples",
    "/evaluate_trace/",
    "/log_eval_results/",
    "/fetch_experiment_run/",
    "/add_to_run_eval_queue/examples",
    "/add_to_run_eval_queue/traces",
    "/save_scorer/",
    "/fetch_scorers/",
    "/scorer_exists/",
    "/upload_custom_scorer/",
    "/datasets/create_for_judgeval/",
    "/datasets/insert_examples_for_judgeval/",
    "/datasets/pull_for_judgeval/",
    "/datasets/pull_all_for_judgeval/",
    "/projects/resolve/",
    "/e2e_fetch_trace/",
    "/e2e_fetch_span_score/",
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
            yield resolved


def filter_schemas() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    processed_schema_names: set[str] = set()
    schemas_to_scan: Any = {
        path: spec_data
        for path, spec_data in SPEC["paths"].items()
        if path in JUDGEVAL_PATHS
    }

    while True:
        to_commit: Dict[str, Any] = {}
        for schema_name in get_referenced_schemas(schemas_to_scan):
            if schema_name in processed_schema_names:
                continue

            assert schema_name in SPEC["components"]["schemas"], (
                f"Schema {schema_name} not found in components.schemas"
            )

            schema = SPEC["components"]["schemas"][schema_name]
            to_commit[schema_name] = schema
            processed_schema_names.add(schema_name)

        if not to_commit:
            break

        result.update(to_commit)
        schemas_to_scan = to_commit

    return result


filtered_paths = {
    path: spec_data
    for path, spec_data in SPEC["paths"].items()
    if path in JUDGEVAL_PATHS
}

spec = {
    "openapi": SPEC["openapi"],
    "info": SPEC["info"],
    "paths": filtered_paths,
    "components": {
        **SPEC["components"],
        "schemas": filter_schemas(),
    },
}

print(orjson.dumps(spec, option=orjson.OPT_INDENT_2).decode("utf-8"))
