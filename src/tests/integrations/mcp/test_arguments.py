from __future__ import annotations

import json
import re
import time

import pytest

from judgeval.integrations.mcp._arguments import (
    AGENT_ARGUMENT_DESCRIPTIONS,
    AGENT_ARGUMENTS,
    conversation_id_prompt,
    mint_conversation_id,
    read_reported_arguments,
    sanitize_intent,
    schema_with_agent_arguments,
    tool_contract,
    without_agent_arguments,
)

UUID_V7 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class TestMintConversationId:
    def test_is_a_uuid_v7_carrying_the_time(self):
        before = int(time.time() * 1000)
        minted = mint_conversation_id()
        after = int(time.time() * 1000)
        assert UUID_V7.match(minted)
        timestamp = int(minted.replace("-", "")[:12], 16)
        assert before <= timestamp <= after

    def test_ids_are_unique_and_sort_by_time(self):
        minted = [mint_conversation_id(now_ms=1_000 + index) for index in range(50)]
        assert len(set(minted)) == 50
        assert minted == sorted(minted)


class TestSanitizeIntent:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Email ann.lee+test@mail.example.co.uk now", "Email [email] now"),
            ("SSN 078-05-1120 on file", "SSN [ssn] on file"),
            ("Card 4111 1111 1111 1111 declined", "Card [card] declined"),
            ("Card 4111-1111-1111-1111 declined", "Card [card] declined"),
            ("Card 4111111111111111.", "Card [card]."),
            # Fails the card checksum, but a long run of digits could be a phone.
            ("Order 1234 5678 9012 3456 shipped", "Order [phone] shipped"),
            ("Server 192.168.0.1 is down", "Server [ip] is down"),
            ("Server fe80::1ff:fe23:4567:890a is down", "Server [ip] is down"),
            ("Hosts 2001:db8::1 and ::1 are down", "Hosts [ip] and [ip] are down"),
            ("Calling Foo::bar and std::vector", "Calling Foo::bar and std::vector"),
            ("Meeting at 10:30 today", "Meeting at 10:30 today"),
            ("Call +1 (415) 555-0100 back", "Call [phone] back"),
            ("Call 415.555.0100 back", "Call [phone] back"),
            ("Version 1.2.3 and 15 items", "Version 1.2.3 and 15 items"),
            ("Checking the refund policy", "Checking the refund policy"),
        ],
    )
    def test_removes_well_defined_identifiers(self, text, expected):
        assert sanitize_intent(text) == expected

    def test_is_capped(self):
        assert len(sanitize_intent("word " * 400)) == 500


class TestToolContract:
    def test_reads_declared_names(self):
        contract = tool_contract(
            {
                "type": "object",
                "properties": {"a": {}, "b": {}},
                "additionalProperties": False,
            }
        )
        assert contract is not None
        assert contract.declared == frozenset({"a", "b"})
        assert contract.open is False

    def test_object_without_properties_is_closed_and_empty(self):
        contract = tool_contract({"type": "object"})
        assert contract is not None
        assert contract.declared == frozenset()
        assert contract.open is False

    @pytest.mark.parametrize("additional", [True, {"type": "string"}])
    def test_accepting_other_keys_is_open(self, additional):
        contract = tool_contract(
            {"type": "object", "properties": {}, "additionalProperties": additional}
        )
        assert contract is not None and contract.open is True

    @pytest.mark.parametrize(
        "schema",
        [
            None,
            "object",
            {"type": "array"},
            {"properties": {"a": {}}},
            {"type": "object", "oneOf": [{}]},
            {"type": "object", "anyOf": [{}]},
            {"type": "object", "allOf": [{}]},
            {"type": "object", "$ref": "#/$defs/Input"},
        ],
    )
    def test_unsafe_schemas_get_no_contract(self, schema):
        assert tool_contract(schema) is None


class TestReadReportedArguments:
    def test_reads_all_three(self):
        conversation_id = mint_conversation_id()
        reported = read_reported_arguments(
            {
                "a": 1,
                "context": "Refunding a@b.co",
                "llm_model": "gpt-6",
                "conversation_id": conversation_id,
            },
            frozenset({"a"}),
        )
        assert reported.intent == "Refunding [email]"
        assert reported.model == "gpt-6"
        assert reported.conversation_id == conversation_id

    def test_names_the_tool_declares_are_the_tools(self):
        reported = read_reported_arguments(
            {"context": "mine", "llm_model": "m", "conversation_id": "c"},
            frozenset(AGENT_ARGUMENTS),
        )
        assert reported == type(reported)(None, None, None)

    @pytest.mark.parametrize("model", ["unknown", "UNKNOWN", "Unknown", ""])
    def test_unknown_model_is_dropped(self, model):
        assert read_reported_arguments({"llm_model": model}, frozenset()).model is None

    @pytest.mark.parametrize(
        "value",
        [
            "anything",
            "4c7b1c4e-3f4c-4b7e-9f0a-2d5e8c1a7b3d",
            "01a1b883-95fd-7271-ccfc-b8b66546f84e",  # wrong variant
            123,
            None,
        ],
    )
    def test_only_uuid_v7_conversation_ids_are_accepted(self, value):
        reported = read_reported_arguments({"conversation_id": value}, frozenset())
        assert reported.conversation_id is None

    @pytest.mark.parametrize("value", [1, ["x"], {"a": 1}, None, ""])
    def test_non_text_values_are_ignored(self, value):
        reported = read_reported_arguments(
            {"context": value, "llm_model": value}, frozenset()
        )
        assert reported.intent is None and reported.model is None


def test_without_agent_arguments_keeps_the_tools_own():
    arguments = {"a": 1, "context": "c", "llm_model": "m", "conversation_id": "x"}
    assert without_agent_arguments(arguments, frozenset({"a"})) == {"a": 1}
    assert without_agent_arguments(arguments, frozenset({"a", "context"})) == {
        "a": 1,
        "context": "c",
    }
    # The caller's mapping is left alone.
    assert len(arguments) == 4


class TestSchemaWithAgentArguments:
    def test_adds_optional_string_arguments(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        updated = schema_with_agent_arguments(schema)
        assert updated["required"] == ["a"]
        assert updated["additionalProperties"] is False
        assert updated["properties"]["a"] == {"type": "integer"}
        for name, description in AGENT_ARGUMENT_DESCRIPTIONS.items():
            assert updated["properties"][name] == {
                "type": "string",
                "description": description,
            }
        # The original schema is unchanged.
        assert set(schema["properties"]) == {"a"}

    def test_the_tools_own_definition_wins(self):
        own = {"type": "integer", "description": "Mine."}
        updated = schema_with_agent_arguments(
            {"type": "object", "properties": {"context": own}}
        )
        assert updated["properties"]["context"] == own

    def test_schema_without_properties(self):
        updated = schema_with_agent_arguments({"type": "object"})
        assert set(updated["properties"]) == set(AGENT_ARGUMENTS)


def test_conversation_id_prompt_is_compact_json():
    conversation_id = mint_conversation_id()
    text = conversation_id_prompt(conversation_id)
    assert text.startswith('{"conversation_id"')
    assert json.loads(text)["conversation_id"] == conversation_id
    assert " " not in text.split('"note"')[0]
