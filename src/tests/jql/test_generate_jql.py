from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_generator() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "generate_jql.py"
    spec = importlib.util.spec_from_file_location("generate_jql", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_type_handles_openapi_31_nulls_and_refs() -> None:
    generator = load_generator()

    assert generator.python_type({"$ref": "#/components/schemas/TimeSpec"}) == "Any"
    assert (
        generator.python_type(
            {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"type": "null"},
                ]
            }
        )
        == "Union[str, int, None]"
    )
    assert generator.python_type({"type": ["string", "null"]}) == "Optional[str]"


def test_typed_dict_source_respects_required_fields_and_documents_refs() -> None:
    generator = load_generator()
    source = generator.typed_dict_source(
        "PublicExampleResponse",
        {
            "type": "object",
            "required": ["required_value"],
            "properties": {
                "required_value": {"type": "string"},
                "optional_value": {"$ref": "#/components/schemas/TimeSpec"},
            },
        },
    )

    assert "class ExampleResponseRequired(TypedDict):" in source
    assert "required_value: str" in source
    assert "class ExampleResponse(ExampleResponseRequired, total=False):" in source
    assert "optional_value: Any  # OpenAPI $ref: TimeSpec" in source

    compile("from typing import Any, TypedDict\n\n" + source, "<generated>", "exec")
