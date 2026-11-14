"""The optional arguments that let agents report intent, model and conversation.

An MCP server can't see the agent's conversation or why it called a tool, so the
server asks: each listed tool gets optional ``intent``, ``llm_model`` and
``conversation_id`` arguments, and they are removed again before the tool runs.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional


INTENT_ARGUMENT = "intent"
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
        "Optional, for analytics only. Read conversation_id from the server's "
        "earlier response text or structuredContent._judgment and pass it unchanged "
        "as {argument_name} on later calls in this conversation. Omit it "
        "on the first call, and do not invent one."
    ),
}

AGENT_ARGUMENTS: FrozenSet[str] = frozenset(AGENT_ARGUMENT_DESCRIPTIONS)


@dataclass(frozen=True)
class ArgumentNames:
    """An explicit name opts application-owned arguments into telemetry."""

    intent: Optional[str] = None
    model: Optional[str] = None
    conversation_id: Optional[str] = None

    @property
    def names(self) -> tuple[str, str, str]:
        return (
            self.intent if self.intent is not None else INTENT_ARGUMENT,
            self.model if self.model is not None else MODEL_ARGUMENT,
            self.conversation_id
            if self.conversation_id is not None
            else CONVERSATION_ARGUMENT,
        )

    def __post_init__(self) -> None:
        if any(not name.strip() for name in self.names) or len(set(self.names)) != 3:
            raise ValueError(
                "MCP telemetry argument names must be non-empty and distinct"
            )


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
    """Which arguments and structured handoff a tool can safely accept."""

    declared: FrozenSet[str]
    # Accepts unknown keys, so an added argument could reach the tool's own code.
    open: bool
    structured_output: Optional[bool] = None


def mint_conversation_id(now_ms: Optional[int] = None) -> str:
    """Mints an RFC 9562 UUIDv7, the only shape accepted when an agent echoes one."""
    timestamp = int(time.time() * 1000) if now_ms is None else now_ms
    data = bytearray(timestamp.to_bytes(6, "big") + os.urandom(10))
    data[6] = (data[6] & 0x0F) | 0x70
    data[8] = (data[8] & 0x3F) | 0x80
    text = data.hex()
    return f"{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"


def tool_contract(
    schema: Any, names: ArgumentNames = ArgumentNames()
) -> Optional[ToolContract]:
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
            declared.update(name for name in names.names if re.search(pattern, name))
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
    args: Mapping[str, Any],
    declared: FrozenSet[str],
    names: ArgumentNames = ArgumentNames(),
    *,
    injected: bool = True,
) -> ReportedArguments:
    """Reads injected arguments and explicitly opted-in business arguments."""

    def read(name: str, configured: Optional[str]) -> Optional[str]:
        if configured is None and (not injected or name in declared):
            return None
        value = args.get(name)
        if not isinstance(value, str) or not value.strip():
            return None
        return value

    intent_name, model_name, conversation_name = names.names
    intent = read(intent_name, names.intent)
    model = read(model_name, names.model)
    conversation_id = read(conversation_name, names.conversation_id)
    return ReportedArguments(
        intent=None if intent is None else intent[:_MAX_INTENT_LENGTH],
        model=None if model is None or model.lower() == "unknown" else model,
        conversation_id=conversation_id
        if conversation_id is not None
        and (
            (
                names.conversation_id is not None
                and (conversation_name in declared or not injected)
            )
            or _UUID_V7.match(conversation_id)
        )
        else None,
    )


def without_agent_arguments(
    args: Mapping[str, Any],
    declared: FrozenSet[str],
    names: ArgumentNames = ArgumentNames(),
    inject_conversation_id: bool = True,
) -> Dict[str, Any]:
    """Removes injected values so a tool's strict schema still accepts the call."""
    injected_names = names.names if inject_conversation_id else names.names[:2]
    return {
        name: value
        for name, value in args.items()
        if name not in injected_names or name in declared
    }


def schema_with_agent_arguments(
    schema: Mapping[str, Any],
    declared: FrozenSet[str],
    inject_conversation_id: bool = True,
    names: ArgumentNames = ArgumentNames(),
) -> Dict[str, Any]:
    """Adds the optional arguments the tool doesn't already declare."""
    properties = schema.get("properties")
    return {
        **schema,
        "properties": {
            **{
                name: {
                    "type": "string",
                    "description": description.replace("{argument_name}", name),
                }
                for name, (default, description) in zip(
                    names.names, AGENT_ARGUMENT_DESCRIPTIONS.items()
                )
                if name not in declared
                and (inject_conversation_id or default != CONVERSATION_ARGUMENT)
            },
            **(properties if isinstance(properties, Mapping) else {}),
        },
    }


def conversation_id_prompt(
    conversation_id: str, argument_name: str = CONVERSATION_ARGUMENT
) -> str:
    """The handoff shared by text and structured-output clients."""
    return json.dumps(
        {
            CONVERSATION_ARGUMENT: conversation_id,
            "note": f"Pass this {argument_name} on later tool calls in this conversation.",
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
    # Adding a named property changes how clients decode a dictionary result:
    # some construct a model containing only that property and drop real keys.
    if not properties or isinstance(schema.get("additionalProperties"), Mapping):
        return None
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
