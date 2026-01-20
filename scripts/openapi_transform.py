from __future__ import annotations

import orjson
import sys
import hashlib
from typing import Any, Dict
import httpx

spec_file = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:10001/openapi/json"

if spec_file.startswith("http"):
    r = httpx.get(spec_file)
    r.raise_for_status()
    SPEC = r.json()
else:
    with open(spec_file, "rb") as f:
        SPEC = orjson.loads(f.read())


def schema_hash(schema: Dict[str, Any]) -> str:
    """Generate a short hash for a schema to detect duplicates."""
    serialized = orjson.dumps(schema, option=orjson.OPT_SORT_KEYS)
    return hashlib.md5(serialized).hexdigest()[:8]


def generate_schema_name(
    path: str, method: str, location: str, status_code: str = ""
) -> str:
    """Generate a schema name from path, method, and location."""
    # Clean up the path to make a valid identifier
    path_parts = (
        path.strip("/")
        .replace("/", "_")
        .replace("-", "_")
        .replace("{", "")
        .replace("}", "")
    )
    method_cap = method.capitalize()

    if status_code:
        return f"{path_parts}_{method_cap}_{location}_{status_code}"
    return f"{path_parts}_{method_cap}_{location}"


def is_extractable_schema(schema: Dict[str, Any]) -> bool:
    """Check if a schema should be extracted (complex enough to warrant extraction)."""
    if not isinstance(schema, dict):
        return False

    # Skip if it's already a reference
    if "$ref" in schema:
        return False

    # Extract objects with properties
    if schema.get("type") == "object" and "properties" in schema:
        return True

    # Extract arrays with complex item types
    if schema.get("type") == "array" and "items" in schema:
        items = schema["items"]
        if isinstance(items, dict) and items.get("type") == "object":
            return True

    # Extract anyOf/oneOf/allOf
    if any(k in schema for k in ["anyOf", "oneOf", "allOf"]):
        return True

    return False


def extract_schemas_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inline schemas from paths and populate components/schemas."""
    schemas: Dict[str, Any] = {}
    schema_by_hash: Dict[str, str] = {}  # hash -> schema name (for deduplication)

    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method not in ["get", "post", "put", "patch", "delete"]:
                continue

            # Process request body
            request_body = operation.get("requestBody", {})
            content = request_body.get("content", {})

            # We only care about application/json for our purposes
            if "application/json" in content:
                json_content = content["application/json"]
                if "schema" in json_content:
                    schema = json_content["schema"]
                    if is_extractable_schema(schema):
                        h = schema_hash(schema)
                        if h in schema_by_hash:
                            # Reuse existing schema
                            schema_name = schema_by_hash[h]
                        else:
                            schema_name = generate_schema_name(path, method, "Request")
                            schemas[schema_name] = schema
                            schema_by_hash[h] = schema_name

                        # Replace with reference in all content types
                        for content_type in content:
                            content[content_type]["schema"] = {
                                "$ref": f"#/components/schemas/{schema_name}"
                            }

            # Process responses
            responses = operation.get("responses", {})
            for status_code, response in responses.items():
                resp_content = response.get("content", {})

                if "application/json" in resp_content:
                    json_content = resp_content["application/json"]
                    if "schema" in json_content:
                        schema = json_content["schema"]
                        if is_extractable_schema(schema):
                            h = schema_hash(schema)
                            if h in schema_by_hash:
                                # Reuse existing schema
                                schema_name = schema_by_hash[h]
                            else:
                                schema_name = generate_schema_name(
                                    path, method, "Response", status_code
                                )
                                schemas[schema_name] = schema
                                schema_by_hash[h] = schema_name

                            json_content["schema"] = {
                                "$ref": f"#/components/schemas/{schema_name}"
                            }

    return schemas


def extract_nested_schemas(schemas: Dict[str, Any]) -> Dict[str, Any]:
    """Extract nested object schemas from within the already extracted schemas."""
    additional_schemas: Dict[str, Any] = {}
    schema_by_hash: Dict[str, str] = {}

    # Build hash map of existing schemas
    for name, schema in schemas.items():
        h = schema_hash(schema)
        schema_by_hash[h] = name

    def process_schema(schema: Any, parent_name: str, prop_name: str = "") -> Any:
        """Recursively process a schema and extract nested objects."""
        if not isinstance(schema, dict):
            return schema

        # Already a reference
        if "$ref" in schema:
            return schema

        # Process array items
        if schema.get("type") == "array" and "items" in schema:
            items = schema["items"]
            if isinstance(items, dict) and is_extractable_schema(items):
                h = schema_hash(items)
                if h in schema_by_hash:
                    nested_name = schema_by_hash[h]
                else:
                    nested_name = (
                        f"{parent_name}_{prop_name}_Item"
                        if prop_name
                        else f"{parent_name}_Item"
                    )
                    additional_schemas[nested_name] = items
                    schema_by_hash[h] = nested_name
                schema["items"] = {"$ref": f"#/components/schemas/{nested_name}"}
            elif isinstance(items, dict):
                schema["items"] = process_schema(items, parent_name, prop_name)
            return schema

        # Process object properties
        if "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if isinstance(prop_schema, dict):
                    schema["properties"][prop] = process_schema(
                        prop_schema, parent_name, prop
                    )

        # Process anyOf/oneOf/allOf
        for key in ["anyOf", "oneOf", "allOf"]:
            if key in schema:
                schema[key] = [process_schema(s, parent_name, key) for s in schema[key]]

        return schema

    # Process each schema
    for name in list(schemas.keys()):
        schemas[name] = process_schema(schemas[name], name)

    # Add the additional schemas we found
    schemas.update(additional_schemas)

    # Process the new schemas too (one more pass)
    for name in list(additional_schemas.keys()):
        schemas[name] = process_schema(schemas[name], name)

    return schemas


# Extract schemas from paths
extracted_schemas = extract_schemas_from_spec(SPEC)

# Extract nested schemas
extracted_schemas = extract_nested_schemas(extracted_schemas)

# Build the final spec
components = SPEC.get("components", {})
existing_schemas = components.get("schemas", {})
all_schemas = {**existing_schemas, **extracted_schemas}

output_spec = {
    "openapi": SPEC["openapi"],
    "info": SPEC["info"],
    "paths": SPEC["paths"],
    "components": {
        **components,
        "schemas": all_schemas,
    },
}

# Add servers if present
if "servers" in SPEC:
    output_spec["servers"] = SPEC["servers"]

# Add security if present
if "security" in SPEC:
    output_spec["security"] = SPEC["security"]

print(orjson.dumps(output_spec, option=orjson.OPT_INDENT_2).decode("utf-8"))
