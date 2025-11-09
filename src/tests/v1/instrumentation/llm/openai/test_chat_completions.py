import pytest

pytest.importorskip("openai")


class BaseOpenAIChatCompletionsTest:
    def verify_tracing_if_wrapped(
        self, client, mock_processor, expected_model_name="gpt-5-nano"
    ):
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            assert span is not None
            assert span.name == "OPENAI_API_CALL"

    def verify_exception_if_wrapped(self, client, mock_processor):
        if hasattr(client, "_judgment_tracer"):
            span = mock_processor.get_last_ended_span()
            assert span is not None


class TestSyncChatCompletions(BaseOpenAIChatCompletionsTest):
    def test_chat_completions_create(self, sync_client_maybe_wrapped, mock_processor):
        response = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        self.verify_tracing_if_wrapped(sync_client_maybe_wrapped, mock_processor)

    def test_multiple_calls_same_client(
        self, sync_client_maybe_wrapped, mock_processor
    ):
        initial_span_count = len(mock_processor.ended_spans)

        response1 = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        response2 = sync_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

        if hasattr(sync_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 2

            span1 = mock_processor.ended_spans[initial_span_count]
            span2 = mock_processor.ended_spans[initial_span_count + 1]

            assert span1.context.span_id != span2.context.span_id

    def test_invalid_model_name_error(self, sync_client_maybe_wrapped, mock_processor):
        with pytest.raises(Exception):
            sync_client_maybe_wrapped.chat.completions.create(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=1000,
            )

        self.verify_exception_if_wrapped(sync_client_maybe_wrapped, mock_processor)


class TestAsyncChatCompletions(BaseOpenAIChatCompletionsTest):
    @pytest.mark.asyncio
    async def test_chat_completions_create(
        self, async_client_maybe_wrapped, mock_processor
    ):
        response = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert response is not None
        assert response.choices
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        self.verify_tracing_if_wrapped(async_client_maybe_wrapped, mock_processor)

    @pytest.mark.asyncio
    async def test_multiple_calls_same_client(
        self, async_client_maybe_wrapped, mock_processor
    ):
        initial_span_count = len(mock_processor.ended_spans)

        response1 = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'first'"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        response2 = await async_client_maybe_wrapped.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'second'"}],
            max_completion_tokens=1000,
            temperature=1,
        )

        assert response1 is not None
        assert response2 is not None
        assert response1.id != response2.id

        if hasattr(async_client_maybe_wrapped, "_judgment_tracer"):
            assert len(mock_processor.ended_spans) == initial_span_count + 2

            span1 = mock_processor.ended_spans[initial_span_count]
            span2 = mock_processor.ended_spans[initial_span_count + 1]

            assert span1.context.span_id != span2.context.span_id

    @pytest.mark.asyncio
    async def test_invalid_model_name_error(
        self, async_client_maybe_wrapped, mock_processor
    ):
        with pytest.raises(Exception):
            await async_client_maybe_wrapped.chat.completions.create(
                model="invalid-model-name-that-does-not-exist",
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=1000,
            )

        self.verify_exception_if_wrapped(async_client_maybe_wrapped, mock_processor)
