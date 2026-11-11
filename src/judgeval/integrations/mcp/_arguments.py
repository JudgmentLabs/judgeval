"""The optional arguments that let agents report intent, model and conversation.

An MCP server can't see the agent's conversation or why it called a tool, so the
server asks: each listed tool gets optional ``context``, ``llm_model`` and
``conversation_id`` arguments, and they are removed again before the tool runs.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional


INTENT_ARGUMENT = "context"
MODEL_ARGUMENT = "llm_model"
CONVERSATION_ARGUMENT = "conversation_id"
RESPONSE_CONTEXT = "_judgment"
_MAX_INTENT_LENGTH = 500
_UUID_V7 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

AGENT_ARGUMENT_DESCRIPTIONS: Mapping[str, str] = {
    INTENT_ARGUMENT: (
        "Optional, for analytics only. In 15-25 words, describe in general terms "
        "why you are calling this tool. Never include names, contact details, "
        "credentials, or other sensitive details."
    ),
    MODEL_ARGUMENT: (
        "Optional, for analytics only. The exact model identifier you are running "
        'as. Pass "unknown" if you are not certain.'
    ),
    CONVERSATION_ARGUMENT: (
        "Optional, for analytics only. Pass the conversation_id from the server's "
        "earlier response text or structuredContent._judgment, unchanged, on later "
        "calls in this conversation. Omit it "
        "on the first call, and do not invent one."
    ),
}

AGENT_ARGUMENTS: FrozenSet[str] = frozenset(AGENT_ARGUMENT_DESCRIPTIONS)

# Keywords that can accept, require or rule out a property without listing it
# under ``properties``. A schema using one is left as it is.
_UNSUPPORTED_KEYWORDS = (
    "oneOf",
    "anyOf",
    "allOf",
    "$ref",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "dependentRequired",
    "dependencies",
    "propertyNames",
)


@dataclass(frozen=True)
class ReportedArguments:
    """Values an agent reported on a call, kept apart from the tool's own input."""

    intent: Optional[str]
    model: Optional[str]
    conversation_id: Optional[str]


@dataclass(frozen=True)
class ToolContract:
    """What a tool's input schema declares, as far as injection is concerned."""

    declared: FrozenSet[str]
    # Accepts unknown keys, so an added argument could reach the tool's own code.
    open: bool


def mint_conversation_id(now_ms: Optional[int] = None) -> str:
    """Mints an RFC 9562 UUIDv7, the only shape accepted when an agent echoes one."""
    timestamp = int(time.time() * 1000) if now_ms is None else now_ms
    data = bytearray(timestamp.to_bytes(6, "big") + os.urandom(10))
    data[6] = (data[6] & 0x0F) | 0x70
    data[8] = (data[8] & 0x3F) | 0x80
    text = data.hex()
    return f"{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"


def tool_contract(schema: Any) -> Optional[ToolContract]:
    """Reads a listed tool's JSON input schema. None if injection isn't safe.

    An absent ``additionalProperties`` permits undeclared business arguments.
    A name the schema requires or matches with
    ``patternProperties`` is the tool's own, like one under ``properties``.
    """
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        return None
    if any(keyword in schema for keyword in _UNSUPPORTED_KEYWORDS):
        return None
    declared = set(_names(schema.get("properties")))
    required = schema.get("required")
    if isinstance(required, list):
        declared.update(name for name in required if isinstance(name, str))
    for pattern in _names(schema.get("patternProperties")):
        try:
            declared.update(
                name for name in AGENT_ARGUMENTS if re.search(pattern, name)
            )
        except re.error:
            # A pattern Python can't read might match one of them.
            return None
    return ToolContract(
        declared=frozenset(declared),
        open=schema.get("additionalProperties") is not False
        and schema.get("unevaluatedProperties") is not False,
    )


def _names(properties: Any) -> FrozenSet[str]:
    if not isinstance(properties, Mapping):
        return frozenset()
    return frozenset(name for name in properties if isinstance(name, str))


def read_reported_arguments(
    args: Mapping[str, Any], declared: FrozenSet[str]
) -> ReportedArguments:
    """Reads agent-reported values, skipping any name the tool itself declares."""

    def read(name: str) -> Optional[str]:
        value = args.get(name)
        if name in declared or not isinstance(value, str) or value == "":
            return None
        return value

    intent = read(INTENT_ARGUMENT)
    model = read(MODEL_ARGUMENT)
    conversation_id = read(CONVERSATION_ARGUMENT)
    return ReportedArguments(
        intent=None if intent is None else intent[:_MAX_INTENT_LENGTH],
        model=None if model is None or model.lower() == "unknown" else model,
        conversation_id=conversation_id
        if conversation_id is not None and _UUID_V7.match(conversation_id)
        else None,
    )


def without_agent_arguments(
    args: Mapping[str, Any], declared: FrozenSet[str]
) -> Dict[str, Any]:
    """Removes injected values so a tool's strict schema still accepts the call."""
    return {
        name: value
        for name, value in args.items()
        if name not in AGENT_ARGUMENTS or name in declared
    }


def schema_with_agent_arguments(
    schema: Mapping[str, Any], declared: FrozenSet[str]
) -> Dict[str, Any]:
    """Adds the optional arguments the tool doesn't already declare."""
    properties = schema.get("properties")
    return {
        **schema,
        "properties": {
            **{
                name: {"type": "string", "description": description}
                for name, description in AGENT_ARGUMENT_DESCRIPTIONS.items()
                if name not in declared
            },
            **(properties if isinstance(properties, Mapping) else {}),
        },
    }


def conversation_id_prompt(conversation_id: str) -> str:
    """The handoff shared by text and structured-output clients."""
    return json.dumps(
        {
            CONVERSATION_ARGUMENT: conversation_id,
            "note": f"Pass this {CONVERSATION_ARGUMENT} on later tool calls in this conversation.",
        },
        separators=(",", ":"),
    )


def conversation_output_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """Extends simple object outputs without claiming a developer's field."""
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        return None
    if any(
        key in schema
        for key in (
            *_UNSUPPORTED_KEYWORDS,
            "patternProperties",
            "maxProperties",
            "const",
            "enum",
        )
    ):
        return None
    properties = schema.get("properties", {})
    if RESPONSE_CONTEXT in properties or RESPONSE_CONTEXT in schema.get("required", []):
        return None
    return {
        **schema,
        "properties": {
            **properties,
            RESPONSE_CONTEXT: {
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["conversation_id", "note"],
                "additionalProperties": False,
            },
        },
    }


def structured_with_prompt(structured: Any, text: str) -> Any:
    """Preserves business fields and never creates structured output from text."""
    if not isinstance(structured, Mapping) or RESPONSE_CONTEXT in structured:
        return structured
    return {**structured, RESPONSE_CONTEXT: json.loads(text)}
