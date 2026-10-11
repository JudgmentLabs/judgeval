from __future__ import annotations

import asyncio
import contextlib
import copy
import gc
import threading
from unittest.mock import MagicMock, AsyncMock
import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.trace.base_tracer import BaseTracer
from judgeval.instrumentation.llm.llm_openai.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)
from tests.instrumentation.llm.openai.conftest import (
    make_chat_completion,
    make_chunk,
)


class TestSyncNonStreaming:
    def test_creates_span(self, tracer, collecting_exporter, sync_openai_client):
        completion = make_chat_completion()
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        assert any(s.name == "OPENAI_API_CALL" for s in collecting_exporter.spans)

    def test_span_has_llm_kind(self, tracer, collecting_exporter, sync_openai_client):
        completion = make_chat_completion()
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    def test_records_model_name(self, tracer, collecting_exporter, sync_openai_client):
        completion = make_chat_completion(model="gpt-4-turbo")
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4-turbo", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME) == "gpt-4-turbo"
        )

    def test_records_token_usage(self, tracer, collecting_exporter, sync_openai_client):
        completion = make_chat_completion(prompt_tokens=15, completion_tokens=8)
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS)
            == 15
        )
        assert span.attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 8

    def test_error_sets_span_error_status(
        self, tracer, collecting_exporter, sync_openai_client
    ):
        sync_openai_client.chat.completions.create = MagicMock(
            side_effect=RuntimeError("fail")
        )
        wrap_chat_completions_create_sync(sync_openai_client)
        with pytest.raises(RuntimeError):
            sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert span.status.status_code.name == "ERROR"

    def test_returns_result_unchanged(self, tracer, sync_openai_client):
        completion = make_chat_completion()
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        result = sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        assert result is completion

    def test_stainless_raw_response_bypasses_wrap(self, tracer, sync_openai_client):
        completion = make_chat_completion()
        original = MagicMock(return_value=completion)
        sync_openai_client.chat.completions.create = original
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(
            model="gpt-4",
            messages=[],
            extra_headers={"X-Stainless-Raw-Response": "stream"},
        )
        original.assert_called_once()


class TestSyncStreaming:
    def test_streaming_creates_span(
        self, tracer, collecting_exporter, sync_openai_client
    ):
        chunks = [make_chunk(content="Hi"), make_chunk(content=" there")]
        sync_openai_client.chat.completions.create = MagicMock(
            return_value=iter(chunks)
        )
        wrap_chat_completions_create_sync(sync_openai_client)
        stream = sync_openai_client.chat.completions.create(
            model="gpt-4", messages=[], stream=True
        )
        list(stream)
        assert any(s.name == "OPENAI_API_CALL" for s in collecting_exporter.spans)

    def test_streaming_accumulates_content(
        self, tracer, collecting_exporter, sync_openai_client
    ):
        chunks = [make_chunk(content="Hello"), make_chunk(content=" world")]
        sync_openai_client.chat.completions.create = MagicMock(
            return_value=iter(chunks)
        )
        wrap_chat_completions_create_sync(sync_openai_client)
        stream = sync_openai_client.chat.completions.create(
            model="gpt-4", messages=[], stream=True
        )
        list(stream)
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert "Hello" in (span.attributes.get(AttributeKeys.GEN_AI_COMPLETION) or "")


class TestAsyncNonStreaming:
    @pytest.mark.asyncio
    async def test_creates_span(self, tracer, collecting_exporter, async_openai_client):
        completion = make_chat_completion()
        async_openai_client.chat.completions.create = AsyncMock(return_value=completion)
        wrap_chat_completions_create_async(async_openai_client)
        await async_openai_client.chat.completions.create(model="gpt-4", messages=[])
        assert any(s.name == "OPENAI_API_CALL" for s in collecting_exporter.spans)

    @pytest.mark.asyncio
    async def test_records_model_name(
        self, tracer, collecting_exporter, async_openai_client
    ):
        completion = make_chat_completion(model="gpt-4")
        async_openai_client.chat.completions.create = AsyncMock(return_value=completion)
        wrap_chat_completions_create_async(async_openai_client)
        await async_openai_client.chat.completions.create(model="gpt-4", messages=[])
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert span.attributes.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME) == "gpt-4"

    @pytest.mark.asyncio
    async def test_error_sets_error_status(
        self, tracer, collecting_exporter, async_openai_client
    ):
        async_openai_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("fail")
        )
        wrap_chat_completions_create_async(async_openai_client)
        with pytest.raises(RuntimeError):
            await async_openai_client.chat.completions.create(
                model="gpt-4", messages=[]
            )
        span = next(s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL")
        assert span.status.status_code.name == "ERROR"


MESSAGES = [{"role": "user", "content": "hi"}]


def _api_spans(exporter):
    return [s for s in exporter.spans if s.name == "OPENAI_API_CALL"]


class TestSyncStreamObject:
    def test_context_manager_and_response(
        self, tracer, collecting_exporter, sse_sync_openai_client
    ):
        client = sse_sync_openai_client
        wrap_chat_completions_create_sync(client)
        with client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        ) as stream:
            assert stream.response.status_code == 200
            text = "".join(
                c.choices[0].delta.content or "" for c in stream if c.choices
            )

        assert text == "Hello"
        spans = _api_spans(collecting_exporter)
        assert len(spans) == 1
        assert spans[0].attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Hello"
        assert spans[0].attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 7

    def test_stream_helper(self, tracer, collecting_exporter, sse_sync_openai_client):
        client = sse_sync_openai_client
        wrap_chat_completions_create_sync(client)
        with client.chat.completions.stream(model="gpt-4", messages=MESSAGES) as stream:
            completion = stream.get_final_completion()

        assert completion.choices[0].message.content == "Hello"
        spans = _api_spans(collecting_exporter)
        assert len(spans) == 1
        assert spans[0].attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Hello"

    @pytest.mark.parametrize("how", ["close", "with"])
    def test_closed_unread_stream_ends_span(
        self, tracer, collecting_exporter, sse_sync_openai_client, how
    ):
        client = sse_sync_openai_client
        wrap_chat_completions_create_sync(client)
        if how == "close":
            stream = client.chat.completions.create(
                model="gpt-4", messages=MESSAGES, stream=True
            )
            stream.close()
        else:
            with client.chat.completions.create(
                model="gpt-4", messages=MESSAGES, stream=True
            ):
                pass

        assert len(_api_spans(collecting_exporter)) == 1

    def test_close_after_partial_read_ends_span_once(
        self, tracer, collecting_exporter, sse_sync_openai_client, caplog
    ):
        client = sse_sync_openai_client
        wrap_chat_completions_create_sync(client)
        stream = client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        )
        next(stream)
        stream.close()
        del stream
        gc.collect()

        assert len(_api_spans(collecting_exporter)) == 1
        assert "Calling end() on an ended span" not in caplog.text

    @pytest.mark.parametrize("how", ["create", "partial_read", "stream_helper"])
    def test_dropped_stream_ends_span_once(
        self, tracer, collecting_exporter, sse_sync_openai_client, caplog, how
    ):
        client = sse_sync_openai_client
        wrap_chat_completions_create_sync(client)
        if how in ("create", "partial_read"):
            stream = client.chat.completions.create(
                model="gpt-4", messages=MESSAGES, stream=True
            )
            if how == "partial_read":
                next(stream)
        else:
            with client.chat.completions.stream(
                model="gpt-4", messages=MESSAGES
            ) as stream:
                pass
        del stream
        gc.collect()

        assert len(_api_spans(collecting_exporter)) == 1
        assert "Calling end() on an ended span" not in caplog.text

    def test_close_from_another_thread_stops_reader(
        self, tracer, collecting_exporter, gated_sync_openai_client, caplog
    ):
        client, body = gated_sync_openai_client
        wrap_chat_completions_create_sync(client)
        stream = client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        )
        chunks = []
        reader = threading.Thread(target=lambda: chunks.extend(stream))
        reader.start()
        assert body.reading.wait(timeout=5)
        stream.close()
        body.release.set()
        reader.join(timeout=5)

        assert not reader.is_alive()
        assert len(chunks) == 1
        spans = _api_spans(collecting_exporter)
        assert len(spans) == 1
        assert spans[0].attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Hel"
        assert "ended span" not in caplog.text

    def test_copy_does_not_recurse(self, tracer, sse_sync_openai_client):
        client = sse_sync_openai_client
        wrap_chat_completions_create_sync(client)
        stream = client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        )
        clone = copy.copy(stream)
        assert clone.response is stream.response
        stream.close()

    def test_observed_function_returning_stream(
        self, tracer, collecting_exporter, sse_sync_openai_client
    ):
        client = sse_sync_openai_client
        wrap_chat_completions_create_sync(client)

        @BaseTracer.observe(span_type="agent")
        def ask():
            return client.chat.completions.create(
                model="gpt-4", messages=MESSAGES, stream=True
            )

        with ask() as stream:
            text = "".join(
                c.choices[0].delta.content or "" for c in stream if c.choices
            )

        assert text == "Hello"
        observed = next(s for s in collecting_exporter.spans if s.name == "ask")
        api = _api_spans(collecting_exporter)
        assert len(api) == 1
        assert api[0].parent.span_id == observed.context.span_id
        assert observed.end_time <= api[0].end_time


class TestAsyncStreamObject:
    @pytest.mark.asyncio
    async def test_context_manager_and_response(
        self, tracer, collecting_exporter, sse_async_openai_client
    ):
        client = sse_async_openai_client
        wrap_chat_completions_create_async(client)
        async with await client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        ) as stream:
            assert stream.response.status_code == 200
            text = "".join(
                [c.choices[0].delta.content or "" async for c in stream if c.choices]
            )

        assert text == "Hello"
        spans = _api_spans(collecting_exporter)
        assert len(spans) == 1
        assert spans[0].attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Hello"
        assert spans[0].attributes.get(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS) == 7

    @pytest.mark.asyncio
    async def test_stream_helper(
        self, tracer, collecting_exporter, sse_async_openai_client
    ):
        client = sse_async_openai_client
        wrap_chat_completions_create_async(client)
        async with client.chat.completions.stream(
            model="gpt-4", messages=MESSAGES
        ) as stream:
            completion = await stream.get_final_completion()

        assert completion.choices[0].message.content == "Hello"
        spans = _api_spans(collecting_exporter)
        assert len(spans) == 1
        assert spans[0].attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Hello"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("how", ["close", "with"])
    async def test_closed_unread_stream_ends_span(
        self, tracer, collecting_exporter, sse_async_openai_client, how
    ):
        client = sse_async_openai_client
        wrap_chat_completions_create_async(client)
        if how == "close":
            stream = await client.chat.completions.create(
                model="gpt-4", messages=MESSAGES, stream=True
            )
            await stream.close()
        else:
            async with await client.chat.completions.create(
                model="gpt-4", messages=MESSAGES, stream=True
            ):
                pass

        assert len(_api_spans(collecting_exporter)) == 1

    @pytest.mark.asyncio
    async def test_close_after_partial_read_ends_span_once(
        self, tracer, collecting_exporter, sse_async_openai_client, caplog
    ):
        client = sse_async_openai_client
        wrap_chat_completions_create_async(client)
        stream = await client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        )
        await stream.__anext__()
        await stream.close()
        del stream
        gc.collect()
        for _ in range(3):
            await asyncio.sleep(0)

        assert len(_api_spans(collecting_exporter)) == 1
        assert "Calling end() on an ended span" not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("how", ["aclose", "aclosing"])
    async def test_aclose_unread_stream_ends_span(
        self, tracer, collecting_exporter, sse_async_openai_client, how
    ):
        client = sse_async_openai_client
        wrap_chat_completions_create_async(client)
        stream = await client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        )
        if how == "aclose":
            await stream.aclose()
        else:
            async with contextlib.aclosing(stream):
                pass

        assert len(_api_spans(collecting_exporter)) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("how", ["create", "partial_read", "stream_helper"])
    async def test_dropped_stream_ends_span_once(
        self, tracer, collecting_exporter, sse_async_openai_client, caplog, how
    ):
        client = sse_async_openai_client
        wrap_chat_completions_create_async(client)
        if how in ("create", "partial_read"):
            stream = await client.chat.completions.create(
                model="gpt-4", messages=MESSAGES, stream=True
            )
            if how == "partial_read":
                await stream.__anext__()
        else:
            async with client.chat.completions.stream(
                model="gpt-4", messages=MESSAGES
            ) as stream:
                pass
        del stream
        gc.collect()
        for _ in range(3):
            await asyncio.sleep(0)

        assert len(_api_spans(collecting_exporter)) == 1
        assert "Calling end() on an ended span" not in caplog.text

    @pytest.mark.asyncio
    async def test_close_from_another_task_stops_reader(
        self, tracer, collecting_exporter, gated_async_openai_client, caplog
    ):
        client, body = gated_async_openai_client
        wrap_chat_completions_create_async(client)
        stream = await client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        )

        async def read_all():
            return [chunk async for chunk in stream]

        reader = asyncio.create_task(read_all())
        await asyncio.wait_for(body.reading.wait(), timeout=5)
        await stream.close()
        body.release.set()
        chunks = await asyncio.wait_for(reader, timeout=5)

        assert len(chunks) == 1
        spans = _api_spans(collecting_exporter)
        assert len(spans) == 1
        assert spans[0].attributes.get(AttributeKeys.GEN_AI_COMPLETION) == "Hel"
        assert "ended span" not in caplog.text

    @pytest.mark.asyncio
    async def test_copy_does_not_recurse(self, tracer, sse_async_openai_client):
        client = sse_async_openai_client
        wrap_chat_completions_create_async(client)
        stream = await client.chat.completions.create(
            model="gpt-4", messages=MESSAGES, stream=True
        )
        clone = copy.copy(stream)
        assert clone.response is stream.response
        await stream.close()


class TestWrappingReplaceMethod:
    def test_wrap_replaces_create_method(self, tracer, sync_openai_client):
        original = sync_openai_client.chat.completions.create
        wrap_chat_completions_create_sync(sync_openai_client)
        assert sync_openai_client.chat.completions.create is not original

    def test_wrap_single_span_per_call(
        self, tracer, collecting_exporter, sync_openai_client
    ):
        completion = make_chat_completion()
        sync_openai_client.chat.completions.create = MagicMock(return_value=completion)
        wrap_chat_completions_create_sync(sync_openai_client)
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        sync_openai_client.chat.completions.create(model="gpt-4", messages=[])
        api_spans = [
            s for s in collecting_exporter.spans if s.name == "OPENAI_API_CALL"
        ]
        assert len(api_spans) == 2
