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
        "earlier response, unchanged, on later calls in this conversation. Omit it "
        "on the first call, and do not invent one."
    ),
}

AGENT_ARGUMENTS: FrozenSet[str] = frozenset(AGENT_ARGUMENT_DESCRIPTIONS)

# Schema keywords whose meaning could change if a property were added.
_COMPOSITION_KEYWORDS = ("oneOf", "anyOf", "allOf", "$ref")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# 13 to 19 digits, optionally grouped, without taking the space after them.
_CARD = re.compile(r"\d(?:[ -]?\d){12,18}")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Includes compressed forms such as ``2001:db8::1`` and ``::1``.
_IPV6 = re.compile(
    r"(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])", re.IGNORECASE
)
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d ().-]{7,}\d\b")


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


def sanitize_intent(text: str) -> str:
    """Removes well-defined personal identifiers from free text.

    It can't recognize free-form data such as names, so the argument's
    description also tells agents not to send them.
    """
    text = _EMAIL.sub("[email]", text)
    text = _SSN.sub("[ssn]", text)
    text = _CARD.sub(
        lambda match: "[card]"
        if _is_luhn_valid(re.sub(r"[ -]", "", match.group(0)))
        else match.group(0),
        text,
    )
    text = _IPV4.sub("[ip]", text)
    text = _IPV6.sub("[ip]", text)
    text = _PHONE.sub("[phone]", text)
    return text[:_MAX_INTENT_LENGTH]


def _is_luhn_valid(digits: str) -> bool:
    total = 0
    for index, character in enumerate(reversed(digits)):
        digit = int(character)
        if index % 2 == 0:
            total += digit
        else:
            total += digit * 2 - 9 if digit > 4 else digit * 2
    return total % 10 == 0


def tool_contract(schema: Any) -> Optional[ToolContract]:
    """Reads a listed tool's JSON input schema. None if injection isn't safe."""
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        return None
    if any(keyword in schema for keyword in _COMPOSITION_KEYWORDS):
        return None
    properties = schema.get("properties")
    declared = (
        frozenset(name for name in properties if isinstance(name, str))
        if isinstance(properties, Mapping)
        else frozenset()
    )
    additional = schema.get("additionalProperties")
    return ToolContract(
        declared=declared,
        open=additional is True or isinstance(additional, Mapping),
    )


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
        intent=None if intent is None else sanitize_intent(intent),
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


def schema_with_agent_arguments(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Adds the optional arguments. A tool's own property of the same name wins."""
    properties = schema.get("properties")
    return {
        **schema,
        "properties": {
            **{
                name: {"type": "string", "description": description}
                for name, description in AGENT_ARGUMENT_DESCRIPTIONS.items()
            },
            **(properties if isinstance(properties, Mapping) else {}),
        },
    }


def conversation_id_prompt(conversation_id: str) -> str:
    """The text appended to a tool result so the agent sends the ID next time.

    It goes in ``content`` only: changing ``structuredContent`` could break a
    client validating against an output schema it listed earlier.
    """
    return json.dumps(
        {
            CONVERSATION_ARGUMENT: conversation_id,
            "note": f"Pass this {CONVERSATION_ARGUMENT} on later tool calls in this conversation.",
        },
        separators=(",", ":"),
    )
