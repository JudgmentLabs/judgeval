#!/usr/bin/env python3
"""Generate Python JQL types from public-safe canonical contracts."""

from __future__ import annotations

import argparse
import json
import keyword
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "scripts" / "jql_contract"
OUTPUTS = (
    ROOT / "src" / "judgeval" / "jql" / "_generated_contract.py",
    ROOT / "src" / "judgeval" / "jql" / "_generated_roots.py",
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


def _union_type(alternatives: list[dict[str, Any]]) -> str:
    members = list(dict.fromkeys(python_type(item) for item in alternatives))
    if len(members) == 1:
        return members[0]
    if "None" in members and len(members) == 2:
        member = next(item for item in members if item != "None")
        return f"Optional[{member}]"
    return "Union[" + ", ".join(members) + "]"


def python_type(schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        # Response-schema dependencies are not generated as public classes yet.
        # Keep output importable and annotate the originating field below.
        return "Any"
    alternatives = schema.get("anyOf")
    if alternatives:
        return _union_type(alternatives)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return _union_type([{**schema, "type": item} for item in schema_type])
    if schema_type == "array":
        return f"List[{python_type(schema.get('items', {}))}]"
    if schema_type == "object":
        return "Dict[str, Any]"
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "null": "None",
    }.get(schema_type, "Any")


def schema_references(schema: dict[str, Any]) -> list[str]:
    return sorted(
        {
            reference.rsplit("/", 1)[-1]
            for node in walk(schema)
            if isinstance((reference := node.get("$ref")), str)
        }
    )


def typed_dict_source(name: str, schema: dict[str, Any]) -> str:
    class_name = name.removeprefix("Public")
    required = set(schema.get("required", []))

    def fields(names: list[str]) -> str:
        lines = []
        for field in names:
            field_schema = schema["properties"][field]
            references = schema_references(field_schema)
            comment = f"  # OpenAPI $ref: {', '.join(references)}" if references else ""
            lines.append(f"    {field}: {python_type(field_schema)}{comment}")
        return "\n".join(lines) or "    pass"

    required_fields = [field for field in schema["properties"] if field in required]
    optional_fields = [field for field in schema["properties"] if field not in required]
    if required_fields and optional_fields:
        required_name = f"{class_name}Required"
        return (
            f"class {required_name}(TypedDict):\n{fields(required_fields)}\n\n\n"
            f"class {class_name}({required_name}, total=False):\n"
            f"{fields(optional_fields)}"
        )
    if optional_fields:
        return f"class {class_name}(TypedDict, total=False):\n{fields(optional_fields)}"
    return f"class {class_name}(TypedDict):\n{fields(required_fields)}"


def generate(dal_document: dict[str, Any], public_document: dict[str, Any]) -> None:
    jql_contract = public_jql_contract(dal_document)
    schemas = jql_contract["components"]["schemas"]
    entries = {item["op"] for node in walk(schemas) for item in node.get("x-jql", [])}
    discovery_kinds = schemas["DiscoveryQuery"]["properties"]["kind"]["enum"]
    query_sources = schemas["SourceQuery"]["properties"]["source"]["enum"]

    operation_members = "\n".join(
        f"    {json.dumps(operation)}," for operation in sorted(entries)
    )
    discovery_kind_members = "\n".join(
        f"    {json.dumps(kind)}," for kind in discovery_kinds
    )
    query_source_members = "\n".join(
        f"    {json.dumps(source)}," for source in query_sources
    )
    contract_source = f'''"""Generated from DAL OpenAPI x-jql metadata; do not edit."""

from typing import Literal

SUPPORTED_OPS = (
{operation_members}
)
QUERY_SOURCES = (
{query_source_members}
)
QuerySource = Literal[
{query_source_members}
]
DiscoveryKind = Literal[
{discovery_kind_members}
]
'''
    OUTPUTS[0].write_text(contract_source, encoding="utf-8", newline="\n")

    invalid_sources = [
        source
        for source in query_sources
        if not source.isidentifier() or keyword.iskeyword(source)
    ]
    if invalid_sources:
        raise ValueError(
            f"JQL query sources must be valid Python identifiers: {invalid_sources!r}"
        )
    root_functions = "\n\n\n".join(
        f"""def {source}(filter: Optional[Filter] = None) -> QueryBuilder:
    from judgeval.jql import _query

    return _query({json.dumps(source)}, filter)"""
        for source in query_sources
    )
    root_exports = "\n".join(f'    "{source}",' for source in query_sources)
    roots_source = f'''"""Generated from DAL OpenAPI query sources; do not edit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from judgeval.jql import Filter, QueryBuilder


{root_functions}


__all__ = [
{root_exports}
]
'''
    OUTPUTS[1].write_text(roots_source, encoding="utf-8", newline="\n")

    public_schemas = public_document["components"]["schemas"]
    classes = [
        typed_dict_source(name, public_schemas[name])
        for name in ("PublicJqlQueryResponse", "PublicJqlPresentationResponse")
    ]
    class_source = "\n\n\n".join(classes)
    typing_names = [
        name
        for name in ("Any", "Dict", "List", "Optional", "TypedDict", "Union")
        if name == "TypedDict"
        or (name == "Any" and "Any" in class_source)
        or f"{name}[" in class_source
    ]
    OUTPUTS[2].write_text(
        '"""Generated from Judgeval public JQL OpenAPI; do not edit."""\n\n'
        f"from typing import {', '.join(typing_names)}\n\n\n" + class_source + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Generated {OUTPUTS[0]} with {len(entries)} canonical JQL operations and "
        f"{OUTPUTS[1]} with {len(query_sources)} query roots and {OUTPUTS[2]}."
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

    dal_document = json.loads(dal_path.read_text(encoding="utf-8"))
    public_document = json.loads(public_path.read_text(encoding="utf-8"))

    if args.sync:
        CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
        (CONTRACT_DIR / "jql-ir.openapi.json").write_text(
            json.dumps(public_jql_contract(dal_document), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (CONTRACT_DIR / "public-openapi.json").write_text(
            json.dumps(public_document, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    generate(dal_document, public_document)


if __name__ == "__main__":
    main()
