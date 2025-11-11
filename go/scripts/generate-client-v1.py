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


def to_struct_name(name: str) -> str:
    camel_case = to_camel_case(name)
    return camel_case[0].upper() + camel_case[1:]


def get_method_name_from_path(path: str, method: str) -> str:
    clean_path = path.strip("/").replace("/", "_").replace("-", "_")
    camel_case = to_camel_case(clean_path)
    return (
        camel_case[0].upper() + camel_case[1:]
    )  # Make it PascalCase for exported methods


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


def get_go_type(schema: Dict[str, Any]) -> str:
    if "$ref" in schema:
        return to_struct_name(resolve_ref(schema["$ref"]))

    for union_key in ["anyOf", "oneOf", "allOf"]:
        if union_key in schema:
            union_schemas = schema[union_key]
            types = set()

            for union_schema in union_schemas:
                if union_schema.get("type") == "null":
                    types.add("null")
                else:
                    types.add(get_go_type(union_schema))

            non_null_types = types - {"null"}
            if len(non_null_types) == 1:
                return list(non_null_types)[0]
            else:
                print(
                    f"Union type with multiple non-null types: {non_null_types}",
                    file=sys.stderr,
                )
                return "interface{}"

    schema_type = schema.get("type", "object")
    type_mapping = {
        "string": "string",
        "integer": "int",
        "number": "float64",
        "boolean": "bool",
        "object": "interface{}",
    }

    if schema_type == "array":
        items = schema.get("items", {})
        return f"[]{get_go_type(items)}" if items else "[]interface{}"

    return type_mapping.get(schema_type, "interface{}")


def generate_struct(className: str, schema: Dict[str, Any]) -> str:
    lines = [
        "package models",
        "",
        "import (",
        '    "encoding/json"',
        ")",
        "",
        f"type {className} struct {{",
    ]

    if "properties" in schema:
        for field_name, property_schema in schema["properties"].items():
            go_type = get_go_type(property_schema)
            json_tag = f'json:"{field_name},omitempty"'

            lines.append(f"    {to_struct_name(field_name)} {go_type} `{json_tag}`")

        lines.append("")

    lines.extend(
        [
            '    AdditionalProperties map[string]interface{} `json:"-"`',
            "}",
            "",
            f"func (m *{className}) UnmarshalJSON(data []byte) error {{",
            f"    type Alias {className}",
            "    aux := &struct {",
            "        *Alias",
            "    }{",
            "        Alias: (*Alias)(m),",
            "    }",
            "    if err := json.Unmarshal(data, &aux); err != nil {{",
            "        return err",
            "    }}",
            "    m.AdditionalProperties = make(map[string]interface{})",
            "    if err := json.Unmarshal(data, &m.AdditionalProperties); err != nil {{",
            "        return err",
            "    }}",
            "    return nil",
            "}",
            "",
            f"func (m {className}) MarshalJSON() ([]byte, error) {{",
            f"    type Alias {className}",
            "    aux := &struct {",
            "        *Alias",
            "    }{",
            "        Alias: (*Alias)(&m),",
            "    }",
            "    ",
            "    result := make(map[string]interface{})",
            "    ",
            "    mainBytes, err := json.Marshal(aux)",
            "    if err != nil {{",
            "        return nil, err",
            "    }}",
            "    ",
            "    if err := json.Unmarshal(mainBytes, &result); err != nil {{",
            "        return nil, err",
            "    }}",
            "    ",
            "    for k, v := range m.AdditionalProperties {{",
            "        result[k] = v",
            "    }}",
            "    ",
            "    return json.Marshal(result)",
            "}",
        ]
    )

    return "\n".join(lines)


def generate_method_signature(
    method_name: str,
    request_type: Optional[str],
    query_params: List[Dict[str, Any]],
    response_type: str,
) -> str:
    params = []

    for param in query_params:
        if param["required"]:
            params.append(f"{param['name']} string")

    if request_type:
        params.append(f"payload *models.{request_type}")

    for param in query_params:
        if not param["required"]:
            params.append(f"{param['name']} *string")

    response_type_ref = (
        f"models.{response_type}" if response_type != "interface{}" else response_type
    )
    return f"func (c *Client) {method_name}({', '.join(params)}) (*{response_type_ref}, error) {{"


def generate_method_body(
    method_name: str,
    path: str,
    method: str,
    request_type: Optional[str],
    query_params: List[Dict[str, Any]],
    response_type: str,
) -> str:
    response_type_ref = (
        f"models.{response_type}" if response_type != "interface{}" else response_type
    )
    lines = []

    if query_params:
        lines.append("    queryParams := make(map[string]string)")
        for param in query_params:
            param_name = param["name"]
            if param["required"]:
                lines.append(f'    queryParams["{param_name}"] = {param_name}')
            else:
                lines.extend(
                    [
                        f"    if {param_name} != nil {{",
                        f'        queryParams["{param_name}"] = *{param_name}',
                        "    }",
                    ]
                )

    if query_params:
        lines.append('    url := c.buildURL("' + path + '", queryParams)')
    else:
        lines.append('    url := c.buildURL("' + path + '", nil)')

    if method in ["GET", "DELETE"]:
        lines.extend(
            [
                f'    req, err := http.NewRequest("{method}", url, nil)',
                "    if err != nil {",
                "        return nil, err",
                "    }",
                "    c.setHeaders(req)",
            ]
        )
    else:
        payload_expr = "payload" if request_type else "struct{}{}"
        lines.extend(
            [
                f"    jsonPayload, err := json.Marshal({payload_expr})",
                "    if err != nil {",
                "        return nil, err",
                "    }",
                f'    req, err := http.NewRequest("{method}", url, bytes.NewBuffer(jsonPayload))',
                "    if err != nil {",
                "        return nil, err",
                "    }",
                "    c.setHeaders(req)",
            ]
        )

    lines.extend(
        [
            "    resp, err := c.httpClient.Do(req)",
            "    if err != nil {",
            "        return nil, err",
            "    }",
            "    defer resp.Body.Close()",
            "",
            "    if resp.StatusCode >= 400 {",
            "        body, _ := io.ReadAll(resp.Body)",
            '        return nil, fmt.Errorf("HTTP Error: %d - %s", resp.StatusCode, string(body))',
            "    }",
            "",
            f"    var result {response_type_ref}",
            "    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {",
            "        return nil, err",
            "    }",
            "    return &result, nil",
        ]
    )

    return "\n".join(lines)


def generate_client_class(methods: List[Dict[str, Any]]) -> str:
    lines = [
        "package api",
        "",
        "import (",
        '    "bytes"',
        '    "encoding/json"',
        '    "fmt"',
        '    "io"',
        '    "net/http"',
        '    "net/url"',
        "",
        '    "github.com/JudgmentLabs/judgeval-go/v1/internal/api/models"',
        ")",
        "",
        "type Client struct {",
        "    baseURL        string",
        "    apiKey         string",
        "    organizationID string",
        "    httpClient     *http.Client",
        "}",
        "",
        "func NewClient(baseURL, apiKey, organizationID string) *Client {",
        "    return &Client{",
        "        baseURL:        baseURL,",
        "        apiKey:         apiKey,",
        "        organizationID: organizationID,",
        "        httpClient:     &http.Client{},",
        "    }",
        "}",
        "",
        "func (c *Client) buildURL(path string, queryParams map[string]string) string {",
        "    u, _ := url.Parse(c.baseURL + path)",
        "    if len(queryParams) > 0 {",
        "        q := u.Query()",
        "        for k, v := range queryParams {",
        "            q.Set(k, v)",
        "        }",
        "        u.RawQuery = q.Encode()",
        "    }",
        "    return u.String()",
        "}",
        "",
        "func (c *Client) setHeaders(req *http.Request) {",
        '    req.Header.Set("Content-Type", "application/json")',
        '    req.Header.Set("Authorization", "Bearer "+c.apiKey)',
        '    req.Header.Set("X-Organization-Id", c.organizationID)',
        "}",
        "",
        "func (c *Client) GetBaseURL() string {",
        "    return c.baseURL",
        "}",
        "",
        "func (c *Client) GetAPIKey() string {",
        "    return c.apiKey",
        "}",
        "",
        "func (c *Client) GetOrganizationID() string {",
        "    return c.organizationID",
        "}",
        "",
    ]

    for method_info in methods:
        signature = generate_method_signature(
            method_info["name"],
            method_info["request_type"],
            method_info["query_params"],
            method_info["response_type"],
        )
        lines.append(signature)

        body = generate_method_body(
            method_info["name"],
            method_info["path"],
            method_info["method"],
            method_info["request_type"],
            method_info["query_params"],
            method_info["response_type"],
        )
        lines.append(body)
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def generate_api_files(spec: Dict[str, Any]) -> None:
    used_schemas = find_used_schemas(spec)
    schemas = spec.get("components", {}).get("schemas", {})

    models_dir = "v1/internal/api/models"
    if os.path.exists(models_dir):
        print(f"Clearing existing models directory: {models_dir}", file=sys.stderr)
        shutil.rmtree(models_dir)

    os.makedirs(models_dir, exist_ok=True)

    print("Generating model structs...", file=sys.stderr)
    for schema_name in used_schemas:
        if schema_name in schemas:
            struct_name = to_struct_name(schema_name)
            model_struct = generate_struct(struct_name, schemas[schema_name])

            with open(f"{models_dir}/{struct_name.lower()}.go", "w") as f:
                f.write(model_struct)

            print(f"Generated model: {struct_name}", file=sys.stderr)

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
                        to_struct_name(request_schema) if request_schema else None
                    ),
                    "query_params": query_params,
                    "response_type": (
                        to_struct_name(response_schema)
                        if response_schema
                        else "EvalResults"  # Default response type
                    ),
                }
                methods.append(method_info)

    api_dir = "v1/internal/api"
    os.makedirs(api_dir, exist_ok=True)

    client_class = generate_client_class(methods)
    with open(f"{api_dir}/client.go", "w") as f:
        f.write(client_class)
    print(f"Generated: {api_dir}/client.go", file=sys.stderr)


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
