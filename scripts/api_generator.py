from __future__ import annotations

import orjson
import sys
from typing import Any, Dict, List, Optional
import httpx
import re

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
    "/get_evaluation_status/",
    "/save_scorer/",
    "/fetch_scorer/",
    "/scorer_exists/",
    "/upload_custom_scorer/",
    "/datasets/create_for_judgeval/",
    "/datasets/insert_examples_for_judgeval/",
    "/datasets/pull_for_judgeval/",
    "/projects/resolve/",
    "/e2e_fetch_trace/",
    "/e2e_fetch_span_score/",
    "/e2e_fetch_trace_scorer_span_score/",
]


def resolve_ref(ref: str) -> str:
    assert ref.startswith("#/components/schemas/"), (
        "Reference must start with #/components/schemas/"
    )
    return ref.replace("#/components/schemas/", "")


def to_snake_case(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def get_method_name_from_path(path: str, method: str) -> str:
    return path.strip("/").replace("/", "_").replace("-", "_")


def get_query_parameters(operation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract query parameters from the operation."""
    parameters = operation.get("parameters", [])
    query_params = []

    for param in parameters:
        if param.get("in") == "query":
            param_info = {
                "name": param["name"],
                "required": param.get("required", False),
                "type": param.get("schema", {}).get("type", "str"),
            }
            query_params.append(param_info)

    return query_params


def get_request_schema(operation: Dict[str, Any]) -> Optional[str]:
    request_body = operation.get("requestBody", {})
    if not request_body:
        return None

    content = request_body.get("content", {})
    if "application/json" in content:
        schema = content["application/json"].get("schema", {})
        if "$ref" in schema:
            return resolve_ref(schema["$ref"])

    return None


def get_response_schema(operation: Dict[str, Any]) -> Optional[str]:
    responses = operation.get("responses", {})
    for status_code in ["200", "201"]:
        if status_code in responses:
            response = responses[status_code]
            content = response.get("content", {})
            if "application/json" in content:
                schema = content["application/json"].get("schema", {})
                if "$ref" in schema:
                    return resolve_ref(schema["$ref"])

    return None


def generate_method_signature(
    method_name: str,
    request_type: Optional[str],
    query_params: List[Dict[str, Any]],
    response_type: str,
    is_async: bool = False,
) -> str:
    async_prefix = "async " if is_async else ""

    params = ["self"]

    # Add required query parameters first
    for param in query_params:
        if param["required"]:
            param_name = param["name"]
            param_type = "str"  # Default to str for simplicity
            params.append(f"{param_name}: {param_type}")

    # Add request body parameter if it exists
    if request_type:
        params.append(f"payload: {request_type}")

    # Add optional query parameters last
    for param in query_params:
        if not param["required"]:
            param_name = param["name"]
            param_type = "str"  # Default to str for simplicity
            params.append(f"{param_name}: Optional[{param_type}] = None")

    params_str = ", ".join(params)
    return f"{async_prefix}def {method_name}({params_str}) -> {response_type}:"


def generate_method_body(
    method_name: str,
    path: str,
    method: str,
    request_type: Optional[str],
    query_params: List[Dict[str, Any]],
    is_async: bool = False,
) -> str:
    async_prefix = "await " if is_async else ""

    # Build query parameters dict if they exist
    if query_params:
        query_lines = ["query_params = {}"]
        for param in query_params:
            param_name = param["name"]
            if param["required"]:
                query_lines.append(f"query_params['{param_name}'] = {param_name}")
            else:
                query_lines.append(f"if {param_name} is not None:")
                query_lines.append(f"    query_params['{param_name}'] = {param_name}")
        query_setup = "\n        ".join(query_lines)
        query_param = "query_params"
    else:
        query_setup = ""
        query_param = "{}"

    if method == "GET":
        if query_setup:
            return f'{query_setup}\n        return {async_prefix}self._request(\n            "{method}",\n            url_for("{path}"),\n            {query_param},\n        )'
        else:
            return f'return {async_prefix}self._request(\n            "{method}",\n            url_for("{path}"),\n            {{}},\n        )'
    else:
        if request_type:
            if query_setup:
                return f'{query_setup}\n        return {async_prefix}self._request(\n            "{method}",\n            url_for("{path}"),\n            payload,\n            params={query_param},\n        )'
            else:
                return f'return {async_prefix}self._request(\n            "{method}",\n            url_for("{path}"),\n            payload,\n        )'
        else:
            if query_setup:
                return f'{query_setup}\n        return {async_prefix}self._request(\n            "{method}",\n            url_for("{path}"),\n            {{}},\n            params={query_param},\n        )'
            else:
                return f'return {async_prefix}self._request(\n            "{method}",\n            url_for("{path}"),\n            {{}},\n        )'


def generate_client_class(
    class_name: str, methods: List[Dict[str, Any]], is_async: bool = False
) -> str:
    lines = [f"class {class_name}:"]
    lines.append('    __slots__ = ("api_key", "organization_id", "client")')
    lines.append("")

    lines.append("    def __init__(self, api_key: str, organization_id: str):")
    lines.append("        self.api_key = api_key")
    lines.append("        self.organization_id = organization_id")
    client_type = "httpx.AsyncClient" if is_async else "httpx.Client"
    lines.append(f"        self.client = {client_type}(timeout=30)")
    lines.append("")

    request_method = "async def _request" if is_async else "def _request"
    lines.append(f"    {request_method}(")
    lines.append(
        '        self, method: Literal["POST", "PATCH", "GET", "DELETE"], url: str, payload: Any, params: Optional[Dict[str, Any]] = None'
    )
    lines.append("    ) -> Any:")
    lines.append('        if method == "GET":')
    lines.append("            r = self.client.request(")
    lines.append("                method,")
    lines.append("                url,")
    lines.append("                params=payload if params is None else params,")
    lines.append(
        "                headers=_headers(self.api_key, self.organization_id),"
    )
    lines.append("            )")
    lines.append("        else:")
    lines.append("            r = self.client.request(")
    lines.append("                method,")
    lines.append("                url,")
    lines.append("                json=json_encoder(payload),")
    lines.append("                params=params,")
    lines.append(
        "                headers=_headers(self.api_key, self.organization_id),"
    )
    lines.append("            )")
    if is_async:
        lines.append("        return _handle_response(await r)")
    else:
        lines.append("        return _handle_response(r)")
    lines.append("")

    for method_info in methods:
        method_name = method_info["name"]
        path = method_info["path"]
        http_method = method_info["method"]
        request_type = method_info["request_type"]
        query_params = method_info["query_params"]
        response_type = method_info["response_type"]

        signature = generate_method_signature(
            method_name, request_type, query_params, response_type, is_async
        )
        lines.append(f"    {signature}")

        body = generate_method_body(
            method_name, path, http_method, request_type, query_params, is_async
        )
        lines.append(f"        {body}")
        lines.append("")

    return "\n".join(lines)


def generate_api_file() -> str:
    lines = [
        "from typing import Dict, Any, Mapping, Literal, Optional",
        "import httpx",
        "from httpx import Response",
        "from judgeval.exceptions import JudgmentAPIError",
        "from judgeval.utils.url import url_for",
        "from judgeval.utils.serialize import json_encoder",
        "from judgeval.api.api_types import *",
        "",
        "",
        "def _headers(api_key: str, organization_id: str) -> Mapping[str, str]:",
        "    return {",
        '        "Content-Type": "application/json",',
        '        "Authorization": f"Bearer {api_key}",',
        '        "X-Organization-Id": organization_id,',
        "    }",
        "",
        "",
        "def _handle_response(r: Response) -> Any:",
        "    if r.status_code >= 400:",
        "        try:",
        '            detail = r.json().get("detail", "")',
        "        except Exception:",
        "            detail = r.text",
        "        raise JudgmentAPIError(r.status_code, detail, r)",
        "    return r.json()",
        "",
        "",
    ]

    filtered_paths = {
        path: spec_data
        for path, spec_data in SPEC["paths"].items()
        if path in JUDGEVAL_PATHS
    }

    for path in JUDGEVAL_PATHS:
        if path not in SPEC["paths"]:
            print(f"Path {path} not found in OpenAPI spec", file=sys.stderr)

    sync_methods = []
    async_methods = []

    for path, path_data in filtered_paths.items():
        for method, operation in path_data.items():
            if method.upper() in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
                method_name = get_method_name_from_path(path, method.upper())
                request_schema = get_request_schema(operation)
                response_schema = get_response_schema(operation)
                query_params = get_query_parameters(operation)

                print(
                    method_name,
                    request_schema,
                    response_schema,
                    query_params,
                    file=sys.stderr,
                )

                if not request_schema:
                    print(f"No request type found for {method_name}", file=sys.stderr)

                if not response_schema:
                    print(
                        f"No response schema found for {method_name}", file=sys.stderr
                    )

                request_type = request_schema if request_schema else None
                response_type = response_schema if response_schema else "Any"

                method_info = {
                    "name": method_name,
                    "path": path,
                    "method": method.upper(),
                    "request_type": request_type,
                    "query_params": query_params,
                    "response_type": response_type,
                }

                sync_methods.append(method_info)
                async_methods.append(method_info)

    sync_client = generate_client_class(
        "JudgmentSyncClient", sync_methods, is_async=False
    )
    async_client = generate_client_class(
        "JudgmentAsyncClient", async_methods, is_async=True
    )

    lines.append(sync_client)
    lines.append("")
    lines.append("")
    lines.append(async_client)
    lines.append("")
    lines.append("")
    lines.append("__all__ = [")
    lines.append('    "JudgmentSyncClient",')
    lines.append('    "JudgmentAsyncClient",')
    lines.append("]")

    return "\n".join(lines)


if __name__ == "__main__":
    api_code = generate_api_file()
    print(api_code)
