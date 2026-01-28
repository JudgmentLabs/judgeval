from __future__ import annotations

import orjson
import sys
from typing import Any, Dict, List, Optional, Set
import httpx
import re

spec_file = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:10001/openapi/json"

if spec_file.startswith("http"):
    r = httpx.get(spec_file)
    r.raise_for_status()
    SPEC = r.json()
else:
    with open(spec_file, "rb") as f:
        SPEC = orjson.loads(f.read())


def normalize_schema_for_comparison(schema: Dict[str, Any]) -> str:
    """
    Create a normalized string representation of a schema for comparison.
    This handles property ordering and optional fields.
    """
    if not isinstance(schema, dict):
        return str(schema)

    # Skip if it's already a reference
    if "$ref" in schema:
        return f"$ref:{schema['$ref']}"

    # Create a normalized copy
    normalized: Dict[str, Any] = {}

    # For objects, normalize properties
    if schema.get("type") == "object" or "properties" in schema:
        normalized["type"] = "object"
        if "properties" in schema:
            # Sort properties by name for consistent comparison
            props = {}
            for key in sorted(schema["properties"].keys()):
                props[key] = normalize_schema_for_comparison(schema["properties"][key])
            normalized["properties"] = props
        if "required" in schema:
            normalized["required"] = sorted(schema["required"])
        if "additionalProperties" in schema:
            normalized["additionalProperties"] = schema["additionalProperties"]
    elif schema.get("type") == "array":
        normalized["type"] = "array"
        if "items" in schema:
            normalized["items"] = normalize_schema_for_comparison(schema["items"])
    elif "anyOf" in schema:
        normalized["anyOf"] = [
            normalize_schema_for_comparison(s) for s in schema["anyOf"]
        ]
    elif "oneOf" in schema:
        normalized["oneOf"] = [
            normalize_schema_for_comparison(s) for s in schema["oneOf"]
        ]
    elif "allOf" in schema:
        normalized["allOf"] = [
            normalize_schema_for_comparison(s) for s in schema["allOf"]
        ]
    else:
        # For primitive types
        for key in ["type", "format", "enum", "const", "default", "minimum", "maximum"]:
            if key in schema:
                normalized[key] = schema[key]

    return orjson.dumps(normalized, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def build_schema_fingerprints(components_schemas: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a mapping of schema fingerprints to schema names.
    Returns: {fingerprint: schema_name}
    """
    fingerprints: Dict[str, str] = {}
    for name, schema in components_schemas.items():
        fingerprint = normalize_schema_for_comparison(schema)
        # If multiple schemas have the same fingerprint, prefer the shorter name
        if fingerprint not in fingerprints or len(name) < len(
            fingerprints[fingerprint]
        ):
            fingerprints[fingerprint] = name
    return fingerprints


def deduplicate_schema(
    schema: Any, fingerprints: Dict[str, str], path: str = ""
) -> Any:
    """
    Recursively walk through a schema and replace inline schemas
    that match component schemas with $ref references.
    """
    if not isinstance(schema, dict):
        return schema

    # Skip if already a reference
    if "$ref" in schema:
        return schema

    # Check if this schema matches a component schema
    fingerprint = normalize_schema_for_comparison(schema)
    if fingerprint in fingerprints:
        schema_name = fingerprints[fingerprint]
        print(
            f"  Deduplicating inline schema at {path} -> {schema_name}", file=sys.stderr
        )
        return {"$ref": f"#/components/schemas/{schema_name}"}

    # Recursively process nested schemas
    result: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            result[key] = {
                prop_name: deduplicate_schema(
                    prop_schema, fingerprints, f"{path}.{prop_name}"
                )
                for prop_name, prop_schema in value.items()
            }
        elif key == "items":
            result[key] = deduplicate_schema(value, fingerprints, f"{path}.items")
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            result[key] = [
                deduplicate_schema(item, fingerprints, f"{path}.{key}[{i}]")
                for i, item in enumerate(value)
            ]
        elif key == "additionalProperties" and isinstance(value, dict):
            result[key] = deduplicate_schema(
                value, fingerprints, f"{path}.additionalProperties"
            )
        else:
            result[key] = value

    return result


def collect_schemas_with_id(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Collect all schemas with $id from the entire OpenAPI spec (including nested schemas).
    Returns a dict mapping schema $id to the schema (without $id field).
    """
    schemas_by_id: Dict[str, Dict[str, Any]] = {}

    def collect_from_value(value: Any) -> None:
        """Recursively walk through the entire spec and collect schemas with $id."""
        if isinstance(value, dict):
            # Check if this dict has an $id (it's a schema)
            if "$id" in value:
                schema_id = value["$id"]
                # Avoid processing the same schema multiple times
                if schema_id not in schemas_by_id:
                    # Store schema without $id field
                    schema_without_id = {k: v for k, v in value.items() if k != "$id"}
                    schemas_by_id[schema_id] = schema_without_id
                    print(f"  Collected schema: {schema_id}", file=sys.stderr)

            # Skip $ref to avoid following references (we only want inline schemas)
            if "$ref" not in value:
                # Recursively process all values in this dict
                for v in value.values():
                    collect_from_value(v)
        elif isinstance(value, list):
            # Recursively process all items in the list
            for item in value:
                collect_from_value(item)
        # For primitive types (str, int, bool, None), do nothing

    # Walk through the entire OpenAPI spec
    print("Collecting schemas with $id from entire OpenAPI spec...", file=sys.stderr)
    collect_from_value(spec)

    print(
        f"Collected {len(schemas_by_id)} schemas with $id: {sorted(schemas_by_id.keys())}",
        file=sys.stderr,
    )
    return schemas_by_id


def to_snake_case(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def to_pascal_case(name: str) -> str:
    """Convert schema name to PascalCase to match datamodel-codegen output."""
    # If name has underscores, split and capitalize each part
    if "_" in name:
        parts = name.split("_")
        return "".join(part[0].upper() + part[1:] if part else "" for part in parts)
    # Otherwise assume it's already PascalCase and return as-is
    return name


def resolve_ref(ref: str) -> str:
    assert ref.startswith("#/components/schemas/"), (
        "Reference must start with #/components/schemas/"
    )
    schema_name = ref.replace("#/components/schemas/", "")
    # Convert to PascalCase to match datamodel-codegen output
    return to_pascal_case(schema_name)


def get_schema_name_from_id(schema: Dict[str, Any]) -> Optional[str]:
    """
    Extract schema name from $id field in inline schema.
    Returns None if schema doesn't have $id or is a reference.
    """
    if "$ref" in schema:
        return resolve_ref(schema["$ref"])
    if "$id" in schema:
        # $id values are already in PascalCase, return as-is
        return schema["$id"]
    return None


def extract_dependencies(
    schema: Dict[str, Any],
    schemas_by_id: Dict[str, Dict[str, Any]],
    visited: Optional[Set[str]] = None,
) -> Set[str]:
    """Extract all schema $id dependencies from a schema."""
    if visited is None:
        visited = set()

    dependencies: Set[str] = set()

    if "$ref" in schema:
        # This shouldn't happen in our case since we're working with inline schemas
        return dependencies

    # If this schema has an $id, we need to get its dependencies (but don't add itself)
    schema_id = schema.get("$id")
    if schema_id and schema_id in visited:
        return dependencies  # Already processed this schema

    if schema_id:
        visited.add(schema_id)
        # Get the full schema from schemas_by_id to extract its dependencies
        if schema_id in schemas_by_id:
            full_schema = schemas_by_id[schema_id]
        else:
            full_schema = schema
    else:
        full_schema = schema

    # Check for nested schemas with $id
    if "properties" in full_schema and isinstance(full_schema["properties"], dict):
        for prop_schema in full_schema["properties"].values():
            if isinstance(prop_schema, dict):
                if "$id" in prop_schema:
                    dep_id = prop_schema["$id"]
                    dependencies.add(dep_id)
                    # Recursively get dependencies of nested schema
                    dependencies.update(
                        extract_dependencies(prop_schema, schemas_by_id, visited)
                    )
                else:
                    dependencies.update(
                        extract_dependencies(prop_schema, schemas_by_id, visited)
                    )

    if "items" in full_schema:
        items_schema = full_schema["items"]
        if isinstance(items_schema, dict):
            if "$id" in items_schema:
                dep_id = items_schema["$id"]
                dependencies.add(dep_id)
                dependencies.update(
                    extract_dependencies(items_schema, schemas_by_id, visited)
                )
            else:
                dependencies.update(
                    extract_dependencies(items_schema, schemas_by_id, visited)
                )

    for union_key in ("anyOf", "oneOf", "allOf"):
        if union_key in full_schema and isinstance(full_schema[union_key], list):
            for item in full_schema[union_key]:
                if isinstance(item, dict):
                    if "$id" in item:
                        dep_id = item["$id"]
                        dependencies.add(dep_id)
                        dependencies.update(
                            extract_dependencies(item, schemas_by_id, visited)
                        )
                    else:
                        dependencies.update(
                            extract_dependencies(item, schemas_by_id, visited)
                        )

    if "additionalProperties" in full_schema and isinstance(
        full_schema["additionalProperties"], dict
    ):
        dependencies.update(
            extract_dependencies(
                full_schema["additionalProperties"], schemas_by_id, visited
            )
        )

    return dependencies


def find_used_schemas(
    spec: Dict[str, Any], schemas_by_id: Dict[str, Dict[str, Any]]
) -> Set[str]:
    """Find all schemas used in request/response bodies."""
    used_schemas = set()

    for path, path_item in spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue

            # Request body
            request_body = operation.get("requestBody", {})
            if request_body:
                for content_type, content in request_body.get("content", {}).items():
                    if "schema" in content:
                        schema = content["schema"]
                        if "$id" in schema:
                            schema_id = schema["$id"]
                            used_schemas.add(schema_id)
                            # Get dependencies
                            if schema_id in schemas_by_id:
                                used_schemas.update(
                                    extract_dependencies(
                                        schemas_by_id[schema_id], schemas_by_id
                                    )
                                )

            # Responses
            for status_code, response in operation.get("responses", {}).items():
                if not isinstance(response, dict):
                    continue
                for content_type, content in response.get("content", {}).items():
                    if "schema" in content:
                        schema = content["schema"]
                        if "$id" in schema:
                            schema_id = schema["$id"]
                            used_schemas.add(schema_id)
                            # Get dependencies
                            if schema_id in schemas_by_id:
                                used_schemas.update(
                                    extract_dependencies(
                                        schemas_by_id[schema_id], schemas_by_id
                                    )
                                )
                        else:
                            # Also check nested schemas even if this one doesn't have $id
                            used_schemas.update(
                                extract_dependencies(schema, schemas_by_id)
                            )

    return used_schemas


def get_python_type(
    schema: Any,
    schemas_by_id: Dict[str, Dict[str, Any]],
    visited: Optional[Set[str]] = None,
) -> str:
    """Convert a schema to a Python type string."""
    if visited is None:
        visited = set()

    # Handle non-dict inputs (lists, primitives, etc.)
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            # If it's a list, process the first item (for tuple validation)
            if schema:
                return f"List[{get_python_type(schema[0], schemas_by_id, visited)}]"
            return "List[Any]"
        # For other non-dict types, return Any
        return "Any"

    if "$ref" in schema:
        return resolve_ref(schema["$ref"])

    if "$id" in schema:
        schema_id = schema["$id"]
        # Return the schema ID directly (it will be a TypedDict class name)
        return schema_id

    # Handle union types (anyOf, oneOf, allOf)
    for union_key in ("anyOf", "oneOf", "allOf"):
        if union_key in schema and isinstance(schema[union_key], list):
            union_types = []
            has_null = False
            for item in schema[union_key]:
                if isinstance(item, dict) and item.get("type") == "null":
                    has_null = True
                else:
                    union_types.append(get_python_type(item, schemas_by_id, visited))

            if not union_types:
                return "Any"

            # Remove duplicates while preserving order
            seen = set()
            unique_types = []
            for t in union_types:
                if t not in seen:
                    seen.add(t)
                    unique_types.append(t)

            if len(unique_types) == 1:
                result = unique_types[0]
            else:
                result = f"Union[{', '.join(unique_types)}]"

            if has_null:
                return f"Optional[{result}]"
            return result

    schema_type = schema.get("type", "object")

    if schema_type == "string":
        return "str"
    elif schema_type == "integer":
        return "int"
    elif schema_type == "number":
        return "float"
    elif schema_type == "boolean":
        return "bool"
    elif schema_type == "array":
        items = schema.get("items", {})
        if items:
            # Handle both dict and list items
            if isinstance(items, dict):
                item_type = get_python_type(items, schemas_by_id, visited)
            elif isinstance(items, list):
                # For tuple validation, use the first item type
                if items:
                    item_type = get_python_type(items[0], schemas_by_id, visited)
                else:
                    item_type = "Any"
            else:
                item_type = "Any"
            return f"List[{item_type}]"
        return "List[Any]"
    elif schema_type == "object" or "properties" in schema:
        # This should be handled by the caller (generate TypedDict)
        return "Dict[str, Any]"
    else:
        return "Any"


def generate_type_definition(
    class_name: str, schema: Dict[str, Any], schemas_by_id: Dict[str, Dict[str, Any]]
) -> str:
    """Generate a type definition from a schema (TypedDict for objects, type alias for arrays)."""
    schema_type = schema.get("type", "object")

    # Handle array types - generate a type alias
    if schema_type == "array":
        items = schema.get("items", {})
        if items:
            item_type = get_python_type(items, schemas_by_id)
            return f"{class_name} = List[{item_type}]"
        return f"{class_name} = List[Any]"

    # Handle object types - generate a TypedDict
    lines = [f"class {class_name}(TypedDict):"]

    required_fields = set(schema.get("required", []))

    if "properties" in schema and isinstance(schema["properties"], dict):
        for field_name, field_schema in schema["properties"].items():
            is_required = field_name in required_fields

            # Get the Python type (this handles nullable and union types)
            # field_schema should be a dict, but handle edge cases
            if not isinstance(field_schema, dict):
                field_type = "Any"
            else:
                field_type = get_python_type(field_schema, schemas_by_id)

            if is_required:
                # If the type is Optional, make it NotRequired as well
                # (nullable fields should be optional to provide)
                if field_type.startswith("Optional["):
                    lines.append(f"    {field_name}: NotRequired[{field_type}]")
                else:
                    lines.append(f"    {field_name}: {field_type}")
            else:
                # Use NotRequired for optional fields
                # If the type is already Optional, we still need NotRequired
                if field_type.startswith("Optional["):
                    lines.append(f"    {field_name}: NotRequired[{field_type}]")
                else:
                    lines.append(
                        f"    {field_name}: NotRequired[Optional[{field_type}]]"
                    )
    else:
        # Empty TypedDict
        lines.append("    pass")

    return "\n".join(lines)


def extract_path_params(path: str) -> List[Dict[str, Any]]:
    """Extract path parameters from OpenAPI path like /projects/{projectId}/datasets"""
    params = []
    for match in re.finditer(r"\{(\w+)\}", path):
        params.append(
            {"name": match.group(1), "required": True, "type": "str", "in": "path"}
        )
    return params


def get_method_name_from_operation(
    operation: Dict[str, Any], path: str, method: str
) -> str:
    """Get method name from operationId, converting camelCase to snake_case."""
    operation_id = operation.get("operationId")
    if operation_id:
        # Convert camelCase operationId to snake_case
        # e.g. "getV1ProjectsByProjectIdDatasets" -> "get_v1_projects_by_project_id_datasets"
        name = to_snake_case(operation_id)
        # Strip v1_ prefix for cleaner method names
        # e.g. "get_v1_projects_by_project_id_datasets" -> "get_projects_by_project_id_datasets"
        name = re.sub(r"^(get|post|put|patch|delete)_v1_", r"\1_", name)
        # Only strip "by_project_id" (the common parent scope)
        # Keep other path params like "by_dataset_name" to distinguish list vs single
        # e.g. "get_projects_by_project_id_datasets" -> "get_project_datasets"
        # but "get_projects_by_project_id_datasets_by_dataset_name" -> "get_project_dataset_by_name"
        name = re.sub(r"_by_project_id", "", name)
        # Replace hyphens with underscores for valid Python identifiers
        name = name.replace("-", "_")
        return name

    # Fallback: construct from path
    name = re.sub(r"\{[^}]+\}", "", path)
    name = name.strip("/").replace("/", "_").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        return "index"
    if name.startswith("v1_"):
        name = name[3:]
    elif name == "v1":
        name = "index"
    return name


def get_query_parameters(operation: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        if schema:
            return get_schema_name_from_id(schema)

    return None


def get_response_schema(operation: Dict[str, Any]) -> Optional[str]:
    responses = operation.get("responses", {})
    for status_code in ["200", "201"]:
        if status_code in responses:
            response = responses[status_code]
            content = response.get("content", {})
            if "application/json" in content:
                schema = content["application/json"].get("schema", {})
                if schema:
                    return get_schema_name_from_id(schema)
            # Also check text/plain for string responses
            elif "text/plain" in content:
                schema = content["text/plain"].get("schema", {})
                if schema:
                    return get_schema_name_from_id(schema)

    return None


def generate_method_signature(
    method_name: str,
    request_type: Optional[str],
    path_params: List[Dict[str, Any]],
    query_params: List[Dict[str, Any]],
    response_type: str,
    is_async: bool = False,
) -> str:
    async_prefix = "async " if is_async else ""

    params = ["self"]

    # Add path params first (always required)
    for param in path_params:
        param_name = to_snake_case(param["name"])
        params.append(f"{param_name}: str")

    # Add required query params
    for param in query_params:
        if param["required"]:
            param_name = param["name"]
            param_type = "str"
            params.append(f"{param_name}: {param_type}")

    # Add payload if present
    if request_type:
        params.append(f"payload: {request_type}")

    # Add optional query params
    for param in query_params:
        if not param["required"]:
            param_name = param["name"]
            param_type = "str"
            params.append(f"{param_name}: Optional[{param_type}] = None")

    params_str = ", ".join(params)
    return f"{async_prefix}def {method_name}({params_str}) -> {response_type}:"


def generate_method_body(
    method_name: str,
    path: str,
    method: str,
    request_type: Optional[str],
    path_params: List[Dict[str, Any]],
    query_params: List[Dict[str, Any]],
    is_async: bool = False,
) -> str:
    async_prefix = "await " if is_async else ""

    # Build URL with path parameter interpolation
    if path_params:
        url_path = path
        for param in path_params:
            snake_name = to_snake_case(param["name"])
            # Replace {paramName} with Python f-string interpolation
            url_path = url_path.replace(f"{{{param['name']}}}", f"{{{snake_name}}}")
        url_expr = f'f"{url_path}"'
    else:
        url_expr = f'"{path}"'

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
            return f'{query_setup}\n        return {async_prefix}self._request(\n            "{method}",\n            url_for({url_expr}, self.base_url),\n            {query_param},\n        )'
        else:
            return f'return {async_prefix}self._request(\n            "{method}",\n            url_for({url_expr}, self.base_url),\n            {{}},\n        )'
    else:
        if request_type:
            if query_setup:
                return f'{query_setup}\n        return {async_prefix}self._request(\n            "{method}",\n            url_for({url_expr}, self.base_url),\n            payload,\n            params={query_param},\n        )'
            else:
                return f'return {async_prefix}self._request(\n            "{method}",\n            url_for({url_expr}, self.base_url),\n            payload,\n        )'
        else:
            if query_setup:
                return f'{query_setup}\n        return {async_prefix}self._request(\n            "{method}",\n            url_for({url_expr}, self.base_url),\n            {{}},\n            params={query_param},\n        )'
            else:
                return f'return {async_prefix}self._request(\n            "{method}",\n            url_for({url_expr}, self.base_url),\n            {{}},\n        )'


def generate_client_class(
    class_name: str, methods: List[Dict[str, Any]], is_async: bool = False
) -> str:
    lines = [f"class {class_name}:"]
    lines.append('    __slots__ = ("base_url", "api_key", "organization_id", "client")')
    lines.append("")

    lines.append(
        "    def __init__(self, base_url: str, api_key: str, organization_id: str):"
    )
    lines.append("        self.base_url = base_url")
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
        path_params = method_info["path_params"]
        query_params = method_info["query_params"]
        response_type = method_info["response_type"]

        signature = generate_method_signature(
            method_name,
            request_type,
            path_params,
            query_params,
            response_type,
            is_async,
        )
        lines.append(f"    {signature}")

        body = generate_method_body(
            method_name,
            path,
            http_method,
            request_type,
            path_params,
            query_params,
            is_async,
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
        "from judgeval.v1.internal.api.api_types import *",
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

    sync_methods = []
    async_methods = []

    # Include endpoints that start with these prefixes
    # - /v1: Main v1 API endpoints (strip v1 from method name)
    # - /otel: OTEL endpoints with their own versioning (keep otel in method name)
    INCLUDE_PREFIXES = ["/v1", "/otel"]

    for path, path_data in SPEC["paths"].items():
        # Only include endpoints that start with our allowed prefixes
        if not any(path.startswith(prefix) for prefix in INCLUDE_PREFIXES):
            continue

        for method, operation in path_data.items():
            if method.upper() in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
                method_name = get_method_name_from_operation(
                    operation, path, method.upper()
                )
                request_schema = get_request_schema(operation)
                response_schema = get_response_schema(operation)
                path_params = extract_path_params(path)
                query_params = get_query_parameters(operation)

                print(
                    method_name,
                    request_schema,
                    response_schema,
                    path_params,
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
                    "path_params": path_params,
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


def generate_api_types() -> None:
    """Generate TypedDict classes from schemas with $id."""
    # Collect all schemas with $id from paths
    schemas_by_id = collect_schemas_with_id(SPEC)

    print(f"Collected {len(schemas_by_id)} schemas with $id", file=sys.stderr)

    # Generate TypedDict classes for ALL collected schemas
    output_path = "src/judgeval/v1/internal/api/api_types.py"
    lines = [
        "from __future__ import annotations",
        "",
        "from typing import TypedDict, Optional, List, Union, Any, Dict",
        "from typing_extensions import NotRequired",
        "",
        "",
    ]

    # Sort schema IDs for consistent output
    generated_count = 0
    for schema_id in sorted(schemas_by_id.keys()):
        schema = schemas_by_id[schema_id]
        class_code = generate_type_definition(schema_id, schema, schemas_by_id)
        lines.append(class_code)
        lines.append("")
        generated_count += 1
        print(f"Generated TypedDict: {schema_id}", file=sys.stderr)

    # Write the file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated {generated_count} TypedDict classes", file=sys.stderr)


if __name__ == "__main__":
    import os

    os.makedirs("src/judgeval/v1/internal/api", exist_ok=True)

    generate_api_types()
    api_code = generate_api_file()

    # Write the client code to file
    client_path = "src/judgeval/v1/internal/api/api_client.py"
    with open(client_path, "w") as f:
        f.write(api_code)
    print(f"Generated API client at {client_path}", file=sys.stderr)
