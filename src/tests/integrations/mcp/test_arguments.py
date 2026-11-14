import re
from dataclasses import asdict

import pytest

from judgeval.integrations.mcp._arguments import (
    ArgumentNames,
    conversation_output_schema,
    conversation_id_prompt,
    mint_conversation_id,
    read_reported_arguments,
    schema_with_agent_arguments,
    structured_with_prompt,
    tool_contract,
    without_agent_arguments,
)


@pytest.mark.parametrize("intent_name", ["intent", "context"])
def test_explicit_argument_names_reuse_owned_fields_and_inject_missing_fields(
    intent_name,
):
    names = ArgumentNames(
        intent=intent_name, model="agent_model", conversation_id="chat_id"
    )
    args = {
        intent_name: "Find tickets",
        "agent_model": "agent-model",
        "chat_id": "customer-chat",
    }
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in args},
    }
    declared = tool_contract(schema, names).declared
    empty = {"type": "object", "properties": {}}
    injected = schema_with_agent_arguments(empty, frozenset(), names=names)
    assert {
        "reused_schema": schema_with_agent_arguments(schema, declared, names=names),
        "reported": asdict(
            read_reported_arguments(args, declared, names, injected=False)
        ),
        "missing": asdict(read_reported_arguments({intent_name: 42}, declared, names)),
        "preserved": without_agent_arguments(args, declared, names),
        "injected_keys": sorted(injected["properties"]),
        "stripped": without_agent_arguments(args, frozenset(), names),
        "disabled_handoff": without_agent_arguments(args, frozenset(), names, False),
        "unverified_echo": read_reported_arguments(
            args, frozenset(), names
        ).conversation_id,
    } == {
        "reused_schema": schema,
        "reported": {
            "intent": "Find tickets",
            "model": "agent-model",
            "conversation_id": "customer-chat",
        },
        "missing": {"intent": None, "model": None, "conversation_id": None},
        "preserved": args,
        "injected_keys": sorted(args),
        "stripped": {},
        "disabled_handoff": {"chat_id": "customer-chat"},
        "unverified_echo": None,
    }


def test_session_ids_are_unique_time_ordered_uuid_v7():
    minted = [mint_conversation_id(now_ms=1000 + index) for index in range(3)]
    pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    assert {
        "valid": all(re.fullmatch(pattern, value) for value in minted),
        "timestamps": [int(value.replace("-", "")[:12], 16) for value in minted],
        "unique": len(set(minted)),
        "ordered": minted == sorted(minted),
    } == {"valid": True, "timestamps": [1000, 1001, 1002], "unique": 3, "ordered": True}


def test_injection_preserves_business_schema_and_input():
    schema = {
        "type": "object",
        "properties": {"intent": {"type": "integer"}},
        "required": ["intent"],
        "additionalProperties": False,
    }
    arguments = {"intent": 42, "llm_model": "m", "conversation_id": "invented"}
    declared = tool_contract(schema).declared
    updated = schema_with_agent_arguments(schema, declared)
    assert {
        "required": updated["required"],
        "additional": updated["additionalProperties"],
        "properties": {
            key: value["type"] for key, value in updated["properties"].items()
        },
        "business": without_agent_arguments(arguments, declared),
        "schema": schema,
        "arguments": arguments,
    } == {
        "required": ["intent"],
        "additional": False,
        "properties": {
            "intent": "integer",
            "llm_model": "string",
            "conversation_id": "string",
        },
        "business": {"intent": 42},
        "schema": {
            "type": "object",
            "properties": {"intent": {"type": "integer"}},
            "required": ["intent"],
            "additionalProperties": False,
        },
        "arguments": {"intent": 42, "llm_model": "m", "conversation_id": "invented"},
    }


@pytest.mark.parametrize(
    "schema, declared, open_schema",
    [
        ({"type": "object"}, set(), True),
        ({"type": "object", "required": ["intent"]}, {"intent"}, True),
        (
            {"type": "object", "patternProperties": {"^con": {}}},
            {"conversation_id"},
            True,
        ),
        ({"type": "object", "additionalProperties": True}, set(), True),
        ({"type": "object", "unevaluatedProperties": {"type": "string"}}, set(), True),
    ],
)
def test_schema_ownership(schema, declared, open_schema):
    contract = tool_contract(schema)
    assert (contract.declared, contract.open) == (frozenset(declared), open_schema)


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {"type": "array"},
        {"properties": {"a": {}}},
        {"type": "object", "oneOf": [{}]},
        {"type": "object", "anyOf": [{}]},
        {"type": "object", "allOf": [{}]},
        {"type": "object", "$ref": "#/Input"},
        {"type": "object", "not": {}},
        {"type": "object", "if": {}},
        {"type": "object", "dependentRequired": {}},
        {"type": "object", "dependentSchemas": {}},
        {"type": "object", "dependencies": {}},
        {"type": "object", "propertyNames": {}},
        {"type": "object", "patternProperties": {"(?<name>x)": {}}},
    ],
)
def test_unsafe_schemas_are_not_injected(schema):
    assert tool_contract(schema) is None


def test_reported_values_preserve_text_and_business_fields_are_not_telemetry():
    session = mint_conversation_id()
    arguments = {
        "intent": "Email a@b.co",
        "llm_model": "gpt-6",
        "conversation_id": session,
    }
    assert {
        "reported": asdict(read_reported_arguments(arguments, frozenset())),
        "owned": asdict(read_reported_arguments(arguments, frozenset(arguments))),
        "bounded": read_reported_arguments({"intent": "a" * 1000}, frozenset()).intent,
    } == {
        "reported": {
            "intent": "Email a@b.co",
            "model": "gpt-6",
            "conversation_id": session,
        },
        "owned": {"intent": None, "model": None, "conversation_id": None},
        "bounded": "a" * 500,
    }


@pytest.mark.parametrize("value", [None, "", 1, ["x"], {"a": 1}])
def test_non_text_telemetry_is_ignored(value):
    arguments = dict.fromkeys(("intent", "llm_model", "conversation_id"), value)
    assert asdict(read_reported_arguments(arguments, frozenset())) == {
        "intent": None,
        "model": None,
        "conversation_id": None,
    }


@pytest.mark.parametrize("model", ["unknown", "UNKNOWN"])
def test_unknown_model_is_omitted(model):
    assert read_reported_arguments({"llm_model": model}, frozenset()).model is None


@pytest.mark.parametrize(
    "session",
    [
        "invented",
        "4c7b1c4e-3f4c-4b7e-9f0a-2d5e8c1a7b3d",
        "01a1b883-95fd-7271-ccfc-b8b66546f84e",
    ],
)
def test_session_ids_require_uuid_v7_and_valid_variant(session):
    assert (
        read_reported_arguments(
            {"conversation_id": session}, frozenset()
        ).conversation_id
        is None
    )


@pytest.mark.parametrize(
    "constraints",
    [
        {"properties": {"_judgment": {"type": "string"}}},
        {"oneOf": []},
        {"maxProperties": 1},
        {"const": {"ok": True}},
        {"enum": [{"ok": True}]},
        {"properties": {}, "additionalProperties": {"type": "integer"}},
        {"additionalProperties": {"type": "integer"}},
    ],
)
def test_incompatible_output_schema_is_preserved(constraints):
    assert (
        conversation_output_schema(
            {"type": "object", "properties": {"ok": {"type": "boolean"}}, **constraints}
        )
        is None
    )


@pytest.mark.parametrize("structured", [None, {"_judgment": "developer-owned"}])
def test_handoff_preserves_text_only_results_and_developer_fields(structured):
    assert (
        structured_with_prompt(structured, conversation_id_prompt("id")) == structured
    )
