#!/usr/bin/env python3
"""Generate Python JQL types from public-safe canonical contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "scripts" / "jql_contract"
OUTPUTS = (
    ROOT / "src" / "judgeval" / "jql" / "_generated_contract.py",
    ROOT / "src" / "judgeval" / "jql" / "_generated_transport.py",
)
ROOT_SCHEMAS = (
    "SourceQuery",
    "DiscoveryQuery",
    "ChartQuery",
    "TableQuery",
    "TimeSpec",
)


def walk(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def public_jql_contract(document: dict[str, Any]) -> dict[str, Any]:
    schemas = document["components"]["schemas"]
    included: dict[str, Any] = {}

    def include_schema(name: str) -> None:
        if name in included:
            return
        schema = schemas.get(name)
        if schema is None:
            raise ValueError(f"Missing DAL schema {name}")
        included[name] = schema
        for node in walk(schema):
            reference = node.get("$ref")
            prefix = "#/components/schemas/"
            if isinstance(reference, str) and reference.startswith(prefix):
                include_schema(reference.removeprefix(prefix))

    for name in ROOT_SCHEMAS:
        include_schema(name)

    return {
        "openapi": "3.1.0",
        "info": {"title": "Public JQL IR", "version": "1"},
        "paths": {},
        "components": {"schemas": included},
    }


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


def generate(dal_document: dict[str, Any], public_document: dict[str, Any]) -> None:
    jql_contract = public_jql_contract(dal_document)
    jql_contract_hash = hashlib.sha256(
        json.dumps(jql_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    public_contract_hash = hashlib.sha256(
        json.dumps(public_document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    schemas = jql_contract["components"]["schemas"]
    entries = {item["op"] for node in walk(schemas) for item in node.get("x-jql", [])}
    discovery_kinds = schemas["DiscoveryQuery"]["properties"]["kind"]["enum"]

    operation_members = "\n".join(
        f"    {json.dumps(operation)}," for operation in sorted(entries)
    )
    literal_members = "\n".join(f"    {json.dumps(kind)}," for kind in discovery_kinds)
    contract_source = f'''"""Generated from DAL OpenAPI x-jql metadata; do not edit."""

from typing import Literal

JQL_IR_SCHEMA_SHA256 = (
    "{jql_contract_hash}"
)
SUPPORTED_OPS = (
{operation_members}
)
DiscoveryKind = Literal[
{literal_members}
]
'''
    OUTPUTS[0].write_text(contract_source)

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
    OUTPUTS[1].write_text(
        '"""Generated from Judgeval public JQL OpenAPI; do not edit."""\n\n'
        f"from typing import {', '.join(typing_names)}\n\n"
        "PUBLIC_JQL_OPENAPI_SHA256 = (\n"
        f'    "{public_contract_hash}"\n'
        ")\n\n\n" + class_source + "\n"
    )
    print(
        f"Generated {OUTPUTS[0]} with {len(entries)} canonical JQL operations and "
        f"{OUTPUTS[1]}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync",
        action="store_true",
        help="refresh the checked-in public-safe contract snapshots",
    )
    parser.add_argument("contracts", nargs="*")
    args = parser.parse_args()
    if args.contracts and len(args.contracts) != 2:
        parser.error("provide both DAL and Judgeval public OpenAPI paths")
    if args.sync and len(args.contracts) != 2:
        parser.error("--sync requires DAL and Judgeval public OpenAPI paths")
    return args


def main() -> None:
    args = parse_args()
    if args.contracts:
        dal_path, public_path = (Path(path).resolve() for path in args.contracts)
    else:
        dal_path = CONTRACT_DIR / "jql-ir.openapi.json"
        public_path = CONTRACT_DIR / "public-openapi.json"

    dal_document = json.loads(dal_path.read_text())
    public_document = json.loads(public_path.read_text())

    if args.sync:
        CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
        (CONTRACT_DIR / "jql-ir.openapi.json").write_text(
            json.dumps(public_jql_contract(dal_document), indent=2) + "\n"
        )
        (CONTRACT_DIR / "public-openapi.json").write_text(
            json.dumps(public_document, indent=2) + "\n"
        )

    generate(dal_document, public_document)


if __name__ == "__main__":
    main()
