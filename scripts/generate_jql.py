#!/usr/bin/env python3
"""Generate Python JQL types from DAL and Judgeval OpenAPI documents."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator


def walk(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def python_type(schema: dict[str, Any]) -> str:
    alternatives = schema.get("anyOf")
    if alternatives:
        non_null = [item for item in alternatives if item.get("type") != "null"]
        if len(non_null) == 1 and len(non_null) != len(alternatives):
            return f"Optional[{python_type(non_null[0])}]"
        return "Union[" + ", ".join(python_type(item) for item in alternatives) + "]"
    if schema.get("type") == "array":
        return f"List[{python_type(schema.get('items', {}))}]"
    if schema.get("type") == "object":
        return "Dict[str, Any]"
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }.get(schema.get("type"), "Any")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: scripts/generate_jql.py <dal-openapi.json> "
            "<judgeval-public-jql-openapi.json>"
        )

    document = json.loads(Path(sys.argv[1]).read_text())
    schemas = document["components"]["schemas"]
    entries = {item["op"] for node in walk(schemas) for item in node.get("x-jql", [])}
    discovery_kinds = schemas["DiscoveryQuery"]["properties"]["kind"]["enum"]

    literal_members = ", ".join(repr(kind) for kind in discovery_kinds)
    rendered = f'''"""Generated from DAL OpenAPI x-jql metadata; do not edit."""

from typing import Literal

SUPPORTED_OPS = {tuple(sorted(entries))!r}
DiscoveryKind = Literal[{literal_members}]
'''
    output = Path("src/judgeval/jql/_generated_contract.py")
    output.write_text(rendered)

    public_document = json.loads(Path(sys.argv[2]).read_text())
    public_schemas = public_document["components"]["schemas"]
    classes = []
    for name in ("PublicJqlQueryResponse", "PublicJqlPresentationResponse"):
        schema = public_schemas[name]
        fields = "\n".join(
            f"    {field}: {python_type(field_schema)}"
            for field, field_schema in schema["properties"].items()
        )
        classes.append(f"class {name.removeprefix('Public')}(TypedDict):\n{fields}")
    class_source = "\n\n\n".join(classes)
    typing_names = ["Any", "Dict", "List", "Optional", "TypedDict"]
    if "Union[" in class_source:
        typing_names.append("Union")
    transport_output = Path("src/judgeval/jql/_generated_transport.py")
    transport_output.write_text(
        '"""Generated from Judgeval public JQL OpenAPI; do not edit."""\n\n'
        f"from typing import {', '.join(typing_names)}\n\n\n" + class_source + "\n"
    )
    print(
        f"Generated {output} with {len(entries)} canonical JQL operations and "
        f"{transport_output}."
    )


if __name__ == "__main__":
    main()
