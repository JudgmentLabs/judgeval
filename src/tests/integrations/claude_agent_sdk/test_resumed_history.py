from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any, List, Optional

from judgeval.integrations.claude_agent_sdk.wrapper import (
    TracingState,
    _create_client_wrapper_class,
    _wrap_query_function,
)
from judgeval.judgment_attribute_keys import AttributeKeys


@dataclasses.dataclass
class TextBlock:
    text: str


@dataclasses.dataclass
class AssistantMessage:
    content: List[Any]
    model: str = "claude-test"


@dataclasses.dataclass
class ResultMessage:
    session_id: str
    usage: Optional[dict] = None


@dataclasses.dataclass
class Options:
    resume: Optional[str] = None
    resume_session_at: Optional[str] = None


async def _fake_query(*, prompt: str, options: Options):
    yield AssistantMessage([TextBlock(f"reply to {prompt}")])
    yield ResultMessage(session_id="session-1")


def test_resumed_query_records_the_earlier_turn(tracer, collecting_exporter):
    query = _wrap_query_function(_fake_query, TracingState())

    async def run(prompt: str, options: Options) -> None:
        async for _ in query(prompt=prompt, options=options):
            pass

    asyncio.run(run("first", Options()))
    asyncio.run(run("second", Options(resume="session-1")))

    llm_inputs = [
        json.loads(span.attributes[AttributeKeys.JUDGMENT_INPUT])
        for span in collecting_exporter.spans
        if span.name == "anthropic.messages.create"
    ]
    first_reply = {
        "content": [{"text": "reply to first", "type": "text"}],
        "role": "assistant",
    }
    assert llm_inputs == [
        [{"content": "first", "role": "user"}],
        [
            {"content": "first", "role": "user"},
            first_reply,
            {"content": "second", "role": "user"},
        ],
    ]


def test_checkpoint_resume_records_no_earlier_turns(tracer, collecting_exporter):
    query = _wrap_query_function(_fake_query, TracingState())

    async def run(prompt: str, options: Options) -> None:
        async for _ in query(prompt=prompt, options=options):
            pass

    asyncio.run(run("first", Options()))
    asyncio.run(run("forked", Options(resume="session-1", resume_session_at="m1")))

    llm_inputs = [
        json.loads(span.attributes[AttributeKeys.JUDGMENT_INPUT])
        for span in collecting_exporter.spans
        if span.name == "anthropic.messages.create"
    ]
    assert llm_inputs[-1] == [{"content": "forked", "role": "user"}]


class _FakeClient:
    sessions = 0

    def __init__(self, options: Optional[Options] = None):
        self.options = options or Options()

    async def connect(self) -> None:
        _FakeClient.sessions += 1
        self.session_id = f"client-session-{_FakeClient.sessions}"

    async def query(self, prompt: str) -> None:
        self.prompt = prompt

    async def receive_response(self):
        yield AssistantMessage([TextBlock(f"reply to {self.prompt}")])
        yield ResultMessage(session_id=self.session_id)


def test_client_history_survives_result_break_and_resets_on_reconnect(
    tracer, collecting_exporter
):
    client = _create_client_wrapper_class(_FakeClient, TracingState())()

    async def ask(prompt: str) -> None:
        await client.query(prompt)
        response = client.receive_response()
        async for message in response:
            if isinstance(message, ResultMessage):
                break
        await response.aclose()

    async def run() -> None:
        await client.connect()
        await ask("first")
        await ask("second")
        await client.connect()
        await ask("fresh")

    asyncio.run(run())

    llm_inputs = [
        json.loads(span.attributes[AttributeKeys.JUDGMENT_INPUT])
        for span in collecting_exporter.spans
        if span.name == "anthropic.messages.create"
    ]
    assert llm_inputs == [
        [{"content": "first", "role": "user"}],
        [
            {"content": "first", "role": "user"},
            {
                "content": [{"text": "reply to first", "type": "text"}],
                "role": "assistant",
            },
            {"content": "second", "role": "user"},
        ],
        [{"content": "fresh", "role": "user"}],
    ]
