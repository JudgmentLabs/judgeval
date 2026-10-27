"""Synthetic SSE over real provider clients, wrappers and span exporters."""

from __future__ import annotations

import asyncio
import copy
import json
import importlib

import httpx
import pytest
from anthropic import Anthropic, AsyncAnthropic
from openai import OpenAI, AsyncOpenAI

from judgeval import wrap
from judgeval.judgment_attribute_keys import AttributeKeys as K


def anthropic_events(kind):
    blocks = []
    if kind == "reasoning":
        blocks.append(
            (
                {"type": "thinking", "thinking": "", "signature": ""},
                [
                    {"type": "thinking_delta", "thinking": "Think "},
                    {"type": "thinking_delta", "thinking": "carefully"},
                    {"type": "signature_delta", "signature": "sig"},
                ],
            )
        )
        blocks.append(({"type": "redacted_thinking", "data": "opaque"}, []))
    blocks.append(
        (
            {"type": "text", "text": ""},
            [
                {"type": "text_delta", "text": "Sun"},
                {"type": "text_delta", "text": "ny"},
            ],
        )
    )
    if kind == "tool":
        for call_id, city in [("call_weather", "SF"), ("call_weather_2", "NY")]:
            blocks.append(
                (
                    {"type": "tool_use", "id": call_id, "name": "weather", "input": {}},
                    [
                        {"type": "input_json_delta", "partial_json": '{"city":'},
                        {"type": "input_json_delta", "partial_json": '"' + city + '"}'},
                    ],
                )
            )
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-actual",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 2,
                },
            },
        }
    ]
    for index, (block, deltas) in enumerate(blocks):
        events.append(
            {"type": "content_block_start", "index": index, "content_block": block}
        )
        events.extend(
            {"type": "content_block_delta", "index": index, "delta": delta}
            for delta in deltas
        )
        events.append({"type": "content_block_stop", "index": index})
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use" if kind == "tool" else "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": 7},
            },
            {"type": "message_stop"},
        ]
    )
    return events


class EventBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, events, error=None):
        self.events = copy.deepcopy(events)
        self.error = error
        self.read = 0
        self.closed = False

    def __iter__(self):
        for event in self.events:
            self.read += 1
            yield (
                f"event: {event.get('type', 'chunk')}\ndata: {json.dumps(event)}\n\n"
            ).encode()
        if self.error:
            raise self.error

    async def __aiter__(self):
        for part in self:
            yield part

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()


def make_client(api, is_async, body):
    cls = (
        (AsyncAnthropic if is_async else Anthropic)
        if api.startswith("anthropic")
        else (AsyncOpenAI if is_async else OpenAI)
    )
    # New client releases use httpx2; keep the same SSE fixtures for both stacks.
    sdk = importlib.import_module(cls.__module__.split(".")[0] + "._base_client")
    http = getattr(sdk, "httpx2", httpx)

    class TransportBody(http.SyncByteStream, http.AsyncByteStream):
        def __iter__(self):
            return iter(body)

        def __aiter__(self):
            return body.__aiter__()

        def close(self):
            body.close()

        async def aclose(self):
            await body.aclose()

    transport = http.MockTransport(
        lambda request: http.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=TransportBody(),
        )
    )
    http_client = (http.AsyncClient if is_async else http.Client)(transport=transport)
    return wrap(cls(api_key="synthetic", http_client=http_client, max_retries=0))


def model_span(exporter):
    spans = [
        s for s in exporter.spans if s.name in ("ANTHROPIC_API_CALL", "OPENAI_API_CALL")
    ]
    assert len(spans) == 1
    return spans[0]


async def consume(client, api, is_async, mode="events", partial=False):
    args = dict(
        model="requested-model",
        messages=[{"role": "user", "content": "Weather?"}],
        max_tokens=100,
    )
    if api == "anthropic-helper":
        if is_async:
            async with client.messages.stream(**args) as stream:
                if mode == "final":
                    return await stream.get_final_message()
                values = []
                async for event in stream.text_stream if mode == "text" else stream:
                    values.append(event)
                    if partial:
                        break
                return values
        else:
            with client.messages.stream(**args) as stream:
                if mode == "final":
                    return stream.get_final_message()
                values = []
                for event in stream.text_stream if mode == "text" else stream:
                    values.append(event)
                    if partial:
                        break
                return values
    if api == "anthropic":
        method = client.messages.create
    elif api == "chat":
        method = client.chat.completions.create
        args.pop("max_tokens")
    else:
        method = client.responses.create
        args = dict(model="requested-model", input="Weather?")
    stream = method(**args, stream=True)
    if is_async:
        stream = await stream
        values = []
        async for event in stream:
            values.append(event)
            if partial:
                await stream.aclose()
                break
    else:
        values = []
        for event in stream:
            values.append(event)
            if partial:
                stream.close()
                break
    return values


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "api,mode",
    [
        ("anthropic", "events"),
        ("anthropic-helper", "events"),
        ("anthropic-helper", "text"),
        ("anthropic-helper", "final"),
    ],
)
@pytest.mark.parametrize("kind", ["text", "tool", "reasoning"])
async def test_anthropic_complete(
    tracer, collecting_exporter, is_async, api, mode, kind
):
    events = anthropic_events(kind)
    body = EventBody(events)
    client = make_client(api, is_async, body)
    result = await consume(client, api, is_async, mode)
    span = model_span(collecting_exporter)
    output = json.loads(span.attributes[K.GEN_AI_COMPLETION])
    assert output["role"] == "assistant"
    assert output["stop_reason"] == ("tool_use" if kind == "tool" else "end_turn")
    blocks = output["content"]
    assert [b["text"] for b in blocks if b["type"] == "text"] == ["Sunny"]
    if kind == "tool":
        calls = [b for b in blocks if b["type"] == "tool_use"]
        assert [(b["id"], b["name"], b["input"]) for b in calls] == [
            ("call_weather", "weather", {"city": "SF"}),
            ("call_weather_2", "weather", {"city": "NY"}),
        ]
        assert all("__json_buf" not in b for b in blocks)
    if kind == "reasoning":
        assert blocks[0] == {
            "type": "thinking",
            "thinking": "Think carefully",
            "signature": "sig",
        }
        assert blocks[1] == {"type": "redacted_thinking", "data": "opaque"}
    assert span.attributes[K.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS] == 12
    assert span.attributes[K.JUDGMENT_USAGE_OUTPUT_TOKENS] == 7
    assert span.attributes[K.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert span.attributes[K.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS] == 2
    assert span.attributes[K.JUDGMENT_LLM_MODEL_NAME] == "claude-actual"
    assert body.read == len(events)
    if api == "anthropic":
        assert [e.type for e in result] == [e["type"] for e in events]
        assert (
            result[0].message.content == []
        )  # Accumulation never mutates application events.
    elif mode == "text":
        assert "".join(result) == "Sunny"
    elif mode == "final":
        assert result.id == output["id"]


def chat_events(kind, finish_reason=None):
    base = dict(
        id="chat_test", object="chat.completion.chunk", model="gpt-actual", created=1
    )
    deltas = [{"role": "assistant", "content": "Sun"}, {"content": "ny"}]
    if kind == "tool":
        deltas.extend(
            [
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":'},
                        }
                        for i in range(2)
                    ]
                },
                {
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": '"NY"}'}},
                        {"index": 0, "function": {"arguments": '"SF"}'}},
                    ]
                },
            ]
        )
    if kind == "refusal":
        deltas = [{"role": "assistant", "refusal": "Cannot "}, {"refusal": "help"}]
    if kind == "reasoning":
        deltas[0].pop("role")
        deltas = [
            {"role": "assistant", "reasoning_content": "Think "},
            {"reasoning_content": "carefully"},
        ] + deltas
    events = [
        {**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
        for d in deltas
    ]
    events.extend(
        [
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                        or ("tool_calls" if kind == "tool" else "stop"),
                    }
                ],
            },
            {
                **base,
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            },
        ]
    )
    return events


def response_events(kind):
    output = [
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Sunny",
                    "annotations": [],
                    "logprobs": None,
                }
            ],
        }
    ]
    if kind == "tool":
        output.extend(
            {
                "type": "function_call",
                "id": f"fc_{i}",
                "call_id": f"call_{i}",
                "name": "weather",
                "arguments": json.dumps({"city": city}),
                "status": "completed",
            }
            for i, city in enumerate(["SF", "NY"])
        )
    if kind == "reasoning":
        output.insert(
            0,
            {
                "id": "rs_test",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Think carefully"}],
                "encrypted_content": "opaque",
            },
        )
    response = dict(
        id="resp_test",
        object="response",
        model="gpt-actual",
        created_at=1,
        status="completed",
        output=output,
        usage={
            "input_tokens": 12,
            "output_tokens": 7,
            "total_tokens": 19,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    )
    events = [
        {
            "type": "response.created",
            "response": {
                **response,
                "status": "in_progress",
                "output": [],
                "usage": None,
            },
        }
    ]
    events.extend(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        }
        for text in ["Sun", "ny"]
    )
    if kind == "tool":
        events.extend(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_0",
                "output_index": 1,
                "delta": part,
            }
            for part in ['{"city":', '"SF"}']
        )
    events.append({"type": "response.completed", "response": response})
    return [{**event, "sequence_number": i} for i, event in enumerate(events)]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("kind", ["text", "tool", "reasoning"])
async def test_openai_complete(tracer, collecting_exporter, is_async, api, kind):
    events = chat_events(kind) if api == "chat" else response_events(kind)
    body = EventBody(events)
    result = await consume(make_client(api, is_async, body), api, is_async)
    span = model_span(collecting_exporter)
    output = json.loads(span.attributes[K.GEN_AI_COMPLETION])
    if api == "chat":
        msg = output["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Sunny"
        if kind == "tool":
            calls = msg["tool_calls"]
            assert all(set(c["function"]) == {"name", "arguments"} for c in calls)
            assert [
                (c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in calls
            ] == [
                ("call_0", "weather", {"city": "SF"}),
                ("call_1", "weather", {"city": "NY"}),
            ]
        if kind == "reasoning":
            assert msg["reasoning_content"] == "Think carefully"
        assert result[0].choices[0].delta.content == (
            None if kind == "reasoning" else "Sun"
        )
    else:
        assert (
            output["output"]
            == result[-1].response.model_dump(mode="json", exclude_unset=True)["output"]
        )
        if kind == "tool":
            assert [
                (c["call_id"], c["name"], json.loads(c["arguments"]))
                for c in output["output"]
                if c["type"] == "function_call"
            ] == [
                ("call_0", "weather", {"city": "SF"}),
                ("call_1", "weather", {"city": "NY"}),
            ]
        if kind == "reasoning":
            assert output["output"][0]["summary"][0]["text"] == "Think carefully"
            assert output["output"][0]["encrypted_content"] == "opaque"
    assert len(result) == len(events)
    assert body.read == len(events)
    assert span.attributes[K.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS] == 9
    assert span.attributes[K.JUDGMENT_USAGE_OUTPUT_TOKENS] == 7
    assert span.attributes[K.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert span.attributes[K.JUDGMENT_LLM_MODEL_NAME] == "gpt-actual"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("api", ["anthropic", "anthropic-helper", "chat", "responses"])
@pytest.mark.parametrize("ending", ["abandon", "truncate", "error", "cancel"])
async def test_unfinished_stream_has_no_completion(
    tracer, collecting_exporter, is_async, api, ending
):
    events = (
        anthropic_events("tool")
        if api.startswith("anthropic")
        else chat_events("tool")
        if api == "chat"
        else response_events("tool")
    )
    if ending != "abandon":
        events = events[:3]
    error = (
        RuntimeError("stream failed")
        if ending == "error"
        else asyncio.CancelledError()
        if ending == "cancel"
        else None
    )
    body = EventBody(events, error=error)
    client = make_client(api, is_async, body)
    if error:
        with pytest.raises(type(error)):
            await consume(client, api, is_async)
    else:
        await consume(client, api, is_async, partial=ending == "abandon")
    assert K.GEN_AI_COMPLETION not in model_span(collecting_exporter).attributes
    assert body.read == (1 if ending == "abandon" else len(events))
    if api == "anthropic-helper":
        assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("finish", ["length", "content_filter", "stop"])
async def test_chat_terminal_finishes(tracer, collecting_exporter, is_async, finish):
    body = EventBody(chat_events("refusal", finish_reason=finish))
    await consume(make_client("chat", is_async, body), "chat", is_async)
    output = json.loads(model_span(collecting_exporter).attributes[K.GEN_AI_COMPLETION])
    assert output["choices"][0]["finish_reason"] == finish
    assert output["choices"][0]["message"]["refusal"] == "Cannot help"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "event_type", ["response.failed", "response.incomplete", "error"]
)
async def test_responses_failure_events(
    tracer, collecting_exporter, is_async, event_type
):
    events = response_events("tool")
    events[-1]["type"] = event_type
    events[-1]["response"]["status"] = (
        "failed" if event_type == "response.failed" else "incomplete"
    )
    body = EventBody(events)
    await consume(make_client("responses", is_async, body), "responses", is_async)
    assert K.GEN_AI_COMPLETION not in model_span(collecting_exporter).attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("api", ["anthropic", "anthropic-helper", "chat", "responses"])
async def test_error_after_terminal_event_does_not_publish(
    tracer, collecting_exporter, is_async, api
):
    events = (
        anthropic_events("text")
        if api.startswith("anthropic")
        else chat_events("text")
        if api == "chat"
        else response_events("text")
    )
    body = EventBody(events, error=RuntimeError("transport failure"))
    with pytest.raises(RuntimeError, match="transport failure"):
        await consume(make_client(api, is_async, body), api, is_async)
    assert K.GEN_AI_COMPLETION not in model_span(collecting_exporter).attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("mode", ["events", "text"])
async def test_helper_caught_error_does_not_publish(
    tracer, collecting_exporter, is_async, mode
):
    body = EventBody(anthropic_events("tool"), error=RuntimeError("transport failure"))
    client = make_client("anthropic-helper", is_async, body)
    args = dict(model="requested", messages=[], max_tokens=100)
    if is_async:
        async with client.messages.stream(**args) as stream:
            with pytest.raises(RuntimeError):
                async for _ in stream.text_stream if mode == "text" else stream:
                    pass
    else:
        with client.messages.stream(**args) as stream:
            with pytest.raises(RuntimeError):
                list(stream.text_stream if mode == "text" else stream)
    span = model_span(collecting_exporter)
    assert K.GEN_AI_COMPLETION not in span.attributes
    assert span.status.status_code.name == "ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_chat_multiple_choices_and_no_usage(
    tracer, collecting_exporter, is_async
):
    events = chat_events("text")[:-1]
    for event in events:
        other = copy.deepcopy(event["choices"][0])
        other["index"] = 1
        if other["delta"].get("content"):
            other["delta"]["content"] = other["delta"]["content"].upper()
        event["choices"].append(other)
    body = EventBody(events)
    client = make_client("chat", is_async, body)
    stream = client.chat.completions.create(
        model="requested",
        messages=[],
        stream=True,
        n=2,
        stream_options={"include_usage": False},
    )
    if is_async:
        stream = await stream
        chunks = [chunk async for chunk in stream]
    else:
        chunks = list(stream)
    assert len(chunks) == len(events)
    output = json.loads(model_span(collecting_exporter).attributes[K.GEN_AI_COMPLETION])
    assert [c["message"]["content"] for c in output["choices"]] == ["Sunny", "SUNNY"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_helper_no_consumption_does_not_drain(
    tracer, collecting_exporter, is_async
):
    body = EventBody(anthropic_events("tool"))
    client = make_client("anthropic-helper", is_async, body)
    if is_async:
        async with client.messages.stream(
            model="requested", messages=[], max_tokens=100
        ):
            pass
    else:
        with client.messages.stream(model="requested", messages=[], max_tokens=100):
            pass
    assert body.read == 0
    assert body.closed
    assert K.GEN_AI_COMPLETION not in model_span(collecting_exporter).attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("api", ["anthropic", "chat"])
async def test_capture_failure_preserves_stream_and_usage(
    tracer, collecting_exporter, monkeypatch, is_async, api
):
    def fail(*args, **kwargs):
        raise ValueError("incompatible SDK accumulator")

    if api == "anthropic":
        monkeypatch.setattr("anthropic.lib.streaming._messages.accumulate_event", fail)
        events = anthropic_events("tool")
    else:
        monkeypatch.setattr(
            "openai.lib.streaming.chat.ChatCompletionStreamState.handle_chunk", fail
        )
        events = chat_events("tool")
    result = await consume(make_client(api, is_async, EventBody(events)), api, is_async)
    assert len(result) == len(events)
    span = model_span(collecting_exporter)
    assert K.GEN_AI_COMPLETION not in span.attributes
    assert span.attributes[K.JUDGMENT_USAGE_OUTPUT_TOKENS] == 7
