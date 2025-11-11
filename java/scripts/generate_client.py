#!/usr/bin/env python3

import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Set
import httpx

JUDGEVAL_PATHS = [
    "/log_eval_results/",
    "/fetch_experiment_run/",
    "/add_to_run_eval_queue/",
    "/get_evaluation_status/",
    "/save_scorer/",
    "/fetch_scorers/",
    "/scorer_exists/",
    "/projects/resolve/",
]

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
SUCCESS_STATUS_CODES = {"200", "201"}
SCHEMA_REF_PREFIX = "#/components/schemas/"


def resolve_ref(ref: str) -> str:
    assert ref.startswith(
        SCHEMA_REF_PREFIX
    ), f"Reference must start with {SCHEMA_REF_PREFIX}"
    return ref.replace(SCHEMA_REF_PREFIX, "")


def to_camel_case(name: str) -> str:
    parts = name.replace("-", "_").split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def to_class_name(name: str) -> str:
    camel_case = to_camel_case(name)
    return camel_case[0].upper() + camel_case[1:]


def get_method_name_from_path(path: str, method: str) -> str:
    clean_path = path.strip("/").replace("/", "_").replace("-", "_")
    return to_camel_case(clean_path)


def get_query_parameters(operation: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "name": param["name"],
            "required": param.get("required", False),
            "type": param.get("schema", {}).get("type", "string"),
        }
        for param in operation.get("parameters", [])
        if param.get("in") == "query"
    ]


def get_schema_from_content(content: Dict[str, Any]) -> Optional[str]:
    if "application/json" in content:
        schema = content["application/json"].get("schema", {})
        return resolve_ref(schema["$ref"]) if "$ref" in schema else None
    return None


def get_request_schema(operation: Dict[str, Any]) -> Optional[str]:
    request_body = operation.get("requestBody", {})
    return (
        get_schema_from_content(request_body.get("content", {}))
        if request_body
        else None
    )


def get_response_schema(operation: Dict[str, Any]) -> Optional[str]:
    responses = operation.get("responses", {})
    for status_code in SUCCESS_STATUS_CODES:
        if status_code in responses:
            result = get_schema_from_content(responses[status_code].get("content", {}))
            if result:
                return result
    return None


def extract_dependencies(
    schema: Dict[str, Any], visited: Optional[Set[str]] = None
) -> Set[str]:
    if visited is None:
        visited = set()

    schema_key = json.dumps(schema, sort_keys=True)
    if schema_key in visited:
        return set()

    visited.add(schema_key)
    dependencies: Set[str] = set()

    if "$ref" in schema:
        return {resolve_ref(schema["$ref"])}

    for key in ["anyOf", "oneOf", "allOf"]:
        if key in schema:
            for s in schema[key]:
                dependencies.update(extract_dependencies(s, visited))

    if "properties" in schema:
        for prop_schema in schema["properties"].values():
            dependencies.update(extract_dependencies(prop_schema, visited))

    if "items" in schema:
        dependencies.update(extract_dependencies(schema["items"], visited))

    if "additionalProperties" in schema and isinstance(
        schema["additionalProperties"], dict
    ):
        dependencies.update(
            extract_dependencies(schema["additionalProperties"], visited)
        )

    return dependencies


def find_used_schemas(spec: Dict[str, Any]) -> Set[str]:
    used_schemas = set()
    schemas = spec.get("components", {}).get("schemas", {})

    for path in JUDGEVAL_PATHS:
        if path in spec["paths"]:
            for method, operation in spec["paths"][path].items():
                if method.upper() in HTTP_METHODS:
                    for schema in [
                        get_request_schema(operation),
                        get_response_schema(operation),
                    ]:
                        if schema:
                            used_schemas.add(schema)

    changed = True
    while changed:
        changed = False
        new_schemas = set()

        for schema_name in used_schemas:
            if schema_name in schemas:
                deps = extract_dependencies(schemas[schema_name])
                for dep in deps:
                    if dep in schemas and dep not in used_schemas:
                        new_schemas.add(dep)
                        changed = True

        used_schemas.update(new_schemas)

    return used_schemas


def get_java_type(schema: Dict[str, Any]) -> str:
    if "$ref" in schema:
        return to_class_name(resolve_ref(schema["$ref"]))

    for union_key in ["anyOf", "oneOf", "allOf"]:
        if union_key in schema:
            union_schemas = schema[union_key]
            types = set()

            for union_schema in union_schemas:
                if union_schema.get("type") == "null":
                    types.add("null")
                else:
                    types.add(get_java_type(union_schema))

            non_null_types = types - {"null"}
            if len(non_null_types) == 1:
                return list(non_null_types)[0]
            else:
                print(
                    f"Union type with multiple non-null types: {non_null_types}",
                    file=sys.stderr,
                )
                return "Object"

    schema_type = schema.get("type", "object")
    type_mapping = {
        "string": "String",
        "integer": "Integer",
        "number": "Double",
        "boolean": "Boolean",
        "object": "Object",
    }

    if schema_type == "array":
        items = schema.get("items", {})
        return f"List<{get_java_type(items)}>" if items else "List<Object>"

    return type_mapping.get(schema_type, "Object")


def generate_model_class(className: str, schema: Dict[str, Any]) -> str:
    required_fields = set(schema.get("required", []))
    has_required = bool(required_fields)

    lines = [
        "package com.judgmentlabs.judgeval.internal.api.models;",
        "",
        "import com.fasterxml.jackson.annotation.JsonAnyGetter;",
        "import com.fasterxml.jackson.annotation.JsonAnySetter;",
        "import com.fasterxml.jackson.annotation.JsonProperty;",
        "import java.util.HashMap;",
        "import java.util.List;",
        "import java.util.Map;",
        "import java.util.Objects;",
    ]

    lines.extend(["", f"public class {className} {{"])

    fields = []
    getters = []
    setters = []
    equals_parts = []
    hashCode_parts = []

    if "properties" in schema:
        for field_name, property_schema in schema["properties"].items():
            java_type = get_java_type(property_schema)
            camel_case_name = to_camel_case(field_name)
            is_required = field_name in required_fields

            field_lines = [
                f'    @JsonProperty("{field_name}")',
                f"    private {java_type} {camel_case_name};",
            ]

            fields.extend(field_lines)

            getters.extend(
                [
                    f"    public {java_type} get{to_class_name(camel_case_name)}() {{",
                    f"        return {camel_case_name};",
                    "    }",
                ]
            )

            setter_param = f"{java_type} {camel_case_name}"
            setters.extend(
                [
                    f"    public void set{to_class_name(camel_case_name)}({setter_param}) {{",
                    f"        this.{camel_case_name} = {camel_case_name};",
                    "    }",
                ]
            )

            equals_parts.append(
                f"Objects.equals({camel_case_name}, other.{camel_case_name})"
            )
            hashCode_parts.append(camel_case_name)

    if fields:
        lines.extend(fields)
        lines.append("")

    lines.extend(
        [
            "    private Map<String, Object> additionalProperties = new HashMap<>();",
            "",
            "    @JsonAnyGetter",
            "    public Map<String, Object> getAdditionalProperties() {",
            "        return additionalProperties;",
            "    }",
            "",
            "    @JsonAnySetter",
            "    public void setAdditionalProperty(String name, Object value) {",
            "        additionalProperties.put(name, value);",
            "    }",
            "",
        ]
    )

    equals_parts.append(
        "Objects.equals(additionalProperties, other.additionalProperties)"
    )
    hashCode_parts.append("Objects.hashCode(additionalProperties)")

    lines.extend(getters)
    lines.append("")
    lines.extend(setters)
    lines.append("")

    lines.extend(
        [
            "    @Override",
            "    public boolean equals(Object obj) {",
            "        if (this == obj) return true;",
            "        if (obj == null || getClass() != obj.getClass()) return false;",
            f"        {className} other = ({className}) obj;",
            f"        return {' && '.join(equals_parts) if equals_parts else 'true'};",
            "    }",
            "",
            "    @Override",
            "    public int hashCode() {",
            f"        return Objects.hash({', '.join(hashCode_parts) if hashCode_parts else ''});",
            "    }",
            "}",
        ]
    )

    return "\n".join(lines)


def generate_method_signature(
    method_name: str,
    request_type: Optional[str],
    query_params: List[Dict[str, Any]],
    response_type: str,
    is_async: bool,
) -> str:
    params = []

    for param in query_params:
        if param["required"]:
            params.append(f"String {param['name']}")

    if request_type:
        params.append(f"{request_type} payload")

    for param in query_params:
        if not param["required"]:
            params.append(f"String {param['name']}")

    return_type = f"CompletableFuture<{response_type}>" if is_async else response_type
    throws_clause = "" if is_async else " throws IOException, InterruptedException"

    return (
        f"    public {return_type} {method_name}({', '.join(params)}){throws_clause} {{"
    )


def generate_method_body(
    method_name: str,
    path: str,
    method: str,
    request_type: Optional[str],
    query_params: List[Dict[str, Any]],
    response_type: str,
    is_async: bool,
) -> str:
    lines = []

    if query_params:
        lines.append("        Map<String, String> queryParams = new HashMap<>();")
        for param in query_params:
            param_name = param["name"]
            if param["required"]:
                lines.append(f'        queryParams.put("{param_name}", {param_name});')
            else:
                param_key = param["name"]
                lines.append(
                    f'        Optional.ofNullable({param_name}).ifPresent(v -> queryParams.put("{param_key}", v));'
                )

    lines.append(
        f'        String url = buildUrl("{path}"{", queryParams" if query_params else ""});'
    )

    if method in ["GET", "DELETE"]:
        lines.extend(
            [
                "        HttpRequest request = HttpRequest.newBuilder()",
                f"                .{method}()",
                "                .uri(URI.create(url))",
                "                .headers(buildHeaders())",
                "                .build();",
            ]
        )
    else:
        payload_expr = "payload" if request_type else "new Object()"

        if is_async:
            lines.extend(
                [
                    "        String jsonPayload;",
                    "        try {",
                    f"            jsonPayload = mapper.writeValueAsString({payload_expr});",
                    "        } catch (Exception e) {",
                    '            throw new RuntimeException("Failed to serialize payload", e);',
                    "        }",
                ]
            )
        else:
            lines.append(
                f"        String jsonPayload = mapper.writeValueAsString({payload_expr});"
            )

        lines.extend(
            [
                "        HttpRequest request = HttpRequest.newBuilder()",
                f"                .{method}(HttpRequest.BodyPublishers.ofString(jsonPayload))",
                "                .uri(URI.create(url))",
                "                .headers(buildHeaders())",
                "                .build();",
            ]
        )

    if is_async:
        lines.extend(
            [
                "        return client.sendAsync(request, HttpResponse.BodyHandlers.ofString())",
                "                .thenApply(this::handleResponse);",
            ]
        )
    else:
        lines.append(
            "        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());"
        )
        if response_type == "Object":
            lines.append("        return handleResponse(response);")
        else:
            lines.append(
                f"        return mapper.readValue(response.body(), {response_type}.class);"
            )

    return "\n".join(lines)


def generate_client_class(
    className: str, methods: List[Dict[str, Any]], is_async: bool
) -> str:
    imports = [
        "package com.judgmentlabs.judgeval.internal.api;",
        "",
        "import com.fasterxml.jackson.core.type.TypeReference;",
        "import com.fasterxml.jackson.databind.ObjectMapper;",
        "import java.io.IOException;",
        "import java.net.URI;",
        "import java.net.http.HttpClient;",
        "import java.net.http.HttpRequest;",
        "import java.net.http.HttpResponse;",
        "import java.util.HashMap;",
        "import java.util.Map;",
        "import java.util.Objects;",
        "import java.util.Optional;",
        "import com.judgmentlabs.judgeval.internal.api.models.*;",
    ]

    if is_async:
        imports.append("import java.util.concurrent.CompletableFuture;")

    lines = imports + [
        "",
        f"public class {className} {{",
        "    private final HttpClient client;",
        "    private final ObjectMapper mapper;",
        "    private final String baseUrl;",
        "    private final String apiKey;",
        "    private final String organizationId;",
        "",
        f"    public {className}(String baseUrl, String apiKey, String organizationId) {{",
        '        this.baseUrl = Objects.requireNonNull(baseUrl, "Base URL cannot be null");',
        '        this.apiKey = Objects.requireNonNull(apiKey, "API key cannot be null");',
        '        this.organizationId = Objects.requireNonNull(organizationId, "Organization ID cannot be null");',
        "        this.client = HttpClient.newBuilder()",
        "                .version(HttpClient.Version.HTTP_1_1)",
        "                .build();",
        "        this.mapper = new ObjectMapper();",
        "    }",
        "",
        "    public String getApiUrl() {",
        "        return baseUrl;",
        "    }",
        "",
        "    public String getApiKey() {",
        "        return apiKey;",
        "    }",
        "",
        "    public String getOrganizationId() {",
        "        return organizationId;",
        "    }",
        "",
        "    private String buildUrl(String path, Map<String, String> queryParams) {",
        "        StringBuilder url = new StringBuilder(baseUrl).append(path);",
        "        if (!queryParams.isEmpty()) {",
        '            url.append("?");',
        "            String queryString = queryParams.entrySet().stream()",
        '                    .map(entry -> entry.getKey() + "=" + entry.getValue())',
        '                    .reduce("", (a, b) -> a.isEmpty() ? b : a + "&" + b);',
        "            url.append(queryString);",
        "        }",
        "        return url.toString();",
        "    }",
        "",
        "    private String buildUrl(String path) {",
        "        return buildUrl(path, new HashMap<>());",
        "    }",
        "",
        "    private String[] buildHeaders() {",
        "        return new String[] {",
        '            "Content-Type",',
        '            "application/json",',
        '            "Authorization",',
        '            "Bearer " + apiKey,',
        '            "X-Organization-Id",',
        "            organizationId",
        "        };",
        "    }",
        "",
    ]

    throws_clause = "" if is_async else " throws IOException"
    lines.append(
        f"    private <T> T handleResponse(HttpResponse<String> response){throws_clause} {{"
    )
    lines.append("        if (response.statusCode() >= 400) {")
    lines.append(
        f'            throw new RuntimeException("HTTP Error: " + response.statusCode() + " - " + response.body());'
    )
    lines.append("        }")
    lines.append("        try {")
    lines.append(
        "            return mapper.readValue(response.body(), new TypeReference<T>() {});"
    )
    lines.append("        } catch (Exception e) {")
    lines.append(
        '            throw new RuntimeException("Failed to parse response", e);'
    )
    lines.append("        }")
    lines.append("    }")
    lines.append("")

    for method_info in methods:
        lines.append(
            generate_method_signature(
                method_info["name"],
                method_info["request_type"],
                method_info["query_params"],
                method_info["response_type"],
                is_async,
            )
        )
        lines.append(
            generate_method_body(
                method_info["name"],
                method_info["path"],
                method_info["method"],
                method_info["request_type"],
                method_info["query_params"],
                method_info["response_type"],
                is_async,
            )
        )
        lines.append("    }")
        lines.append("")

    lines.append("}")
    return "\n".join(lines)


def generate_api_files(spec: Dict[str, Any]) -> None:
    used_schemas = find_used_schemas(spec)
    schemas = spec.get("components", {}).get("schemas", {})

    models_dir = (
        "judgeval-java/src/main/java/com/judgmentlabs/judgeval/internal/api/models"
    )
    if os.path.exists(models_dir):
        print(f"Clearing existing models directory: {models_dir}", file=sys.stderr)
        shutil.rmtree(models_dir)

    os.makedirs(models_dir, exist_ok=True)

    print("Generating model classes...", file=sys.stderr)
    for schema_name in used_schemas:
        if schema_name in schemas:
            className = to_class_name(schema_name)
            model_class = generate_model_class(className, schemas[schema_name])

            with open(f"{models_dir}/{className}.java", "w") as f:
                f.write(model_class)

            print(f"Generated model: {className}", file=sys.stderr)

    filtered_paths = {
        path: spec_data
        for path, spec_data in spec["paths"].items()
        if path in JUDGEVAL_PATHS
    }

    for path in JUDGEVAL_PATHS:
        if path not in spec["paths"]:
            print(f"Path {path} not found in OpenAPI spec", file=sys.stderr)

    methods = []
    for path, path_data in filtered_paths.items():
        for method, operation in path_data.items():
            if method.upper() in HTTP_METHODS:
                method_name = get_method_name_from_path(path, method.upper())
                request_schema = get_request_schema(operation)
                response_schema = get_response_schema(operation)
                query_params = get_query_parameters(operation)

                print(
                    f"{method_name} {request_schema} {response_schema} {query_params}",
                    file=sys.stderr,
                )

                method_info = {
                    "name": method_name,
                    "path": path,
                    "method": method.upper(),
                    "request_type": (
                        to_class_name(request_schema) if request_schema else None
                    ),
                    "query_params": query_params,
                    "response_type": (
                        to_class_name(response_schema) if response_schema else "Object"
                    ),
                }
                methods.append(method_info)

    api_dir = "judgeval-java/src/main/java/com/judgmentlabs/judgeval/internal/api"
    os.makedirs(api_dir, exist_ok=True)

    for is_async, class_name in [
        (False, "JudgmentSyncClient"),
        (True, "JudgmentAsyncClient"),
    ]:
        client_class = generate_client_class(class_name, methods, is_async)
        with open(f"{api_dir}/{class_name}.java", "w") as f:
            f.write(client_class)
        print(f"Generated: {api_dir}/{class_name}.java", file=sys.stderr)


def main():
    spec_file = (
        sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/openapi.json"
    )

    try:
        if spec_file.startswith("http"):
            with httpx.Client() as client:
                response = client.get(spec_file)
                response.raise_for_status()
                spec = response.json()
        else:
            with open(spec_file, "r") as f:
                spec = json.load(f)

        generate_api_files(spec)

    except Exception as e:
        print(f"Error generating API client: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
