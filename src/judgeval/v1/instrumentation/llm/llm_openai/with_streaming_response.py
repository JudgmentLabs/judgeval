from __future__ import annotations
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    AsyncGenerator,
    Iterator,
)

from opentelemetry.trace import Status, StatusCode
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    mutable_wrap_sync,
    immutable_wrap_sync_iterator,
    immutable_wrap_async_iterator,
)
from judgeval.v1.instrumentation.llm.llm_openai.utils import (
    openai_tokens_converter,
    set_cost_attribute,
)

if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer
    from openai import OpenAI, AsyncOpenAI
    from openai._response import APIResponse, AsyncAPIResponse


def wrap_with_streaming_response_sync(tracer: BaseTracer, client: OpenAI) -> None:
    original_func = client.chat.completions.with_streaming_response.create

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_hook(ctx: Dict[str, Any], result: Any) -> Any:
        original_cm = result

        class WrappedResponseContextManager:
            def __init__(self, cm: Any) -> None:
                self._cm = cm

            def __enter__(self) -> Any:
                response = self._cm.__enter__()
                return _wrap_response(ctx, response)

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
                try:
                    return self._cm.__exit__(exc_type, exc_val, exc_tb)
                finally:
                    _finalize_span(ctx, exc_val)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cm, name)

        def _wrap_response(context: Dict[str, Any], response: APIResponse[Any]) -> Any:
            original_iter_lines = response.iter_lines

            def traced_iter_lines() -> Generator[str, None, None]:
                for line in original_iter_lines():
                    yield line

            def yield_hook(inner_ctx: Dict[str, Any], line: str) -> None:
                if not line.startswith("data: ") or line == "data: [DONE]":
                    return
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    return
                if not isinstance(data, dict):
                    return
                _process_chunk_dict(context, data)

            wrapped_iter_lines = immutable_wrap_sync_iterator(
                traced_iter_lines,
                yield_hook=yield_hook,
            )

            class WrappedAPIResponse:
                def __init__(self, resp: APIResponse[Any]) -> None:
                    self._response = resp

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._response, name)

                def iter_lines(self) -> Iterator[str]:
                    return wrapped_iter_lines()

                def parse(self, *, to: type | None = None) -> Any:
                    result = (
                        self._response.parse(to=to) if to else self._response.parse()
                    )
                    if hasattr(result, "__iter__") and hasattr(result, "response"):
                        return _wrap_stream(context, result)
                    _process_completion(context, result)
                    return result

                def json(self) -> object:
                    result = self._response.json()
                    if isinstance(result, dict):
                        _process_json(context, result)
                    return result

                def read(self) -> bytes:
                    return self._response.read()

                def text(self) -> str:
                    return self._response.text()

                def iter_text(self, chunk_size: int | None = None) -> Any:
                    return self._response.iter_text(chunk_size)

                def iter_bytes(self, chunk_size: int | None = None) -> Any:
                    return self._response.iter_bytes(chunk_size)

                def close(self) -> None:
                    return self._response.close()

            return WrappedAPIResponse(response)

        def _wrap_stream(context: Dict[str, Any], stream: Any) -> Any:
            original_iter = stream.__iter__

            def traced_iter() -> Generator[Any, None, None]:
                for chunk in original_iter():
                    yield chunk

            def yield_hook(inner_ctx: Dict[str, Any], chunk: Any) -> None:
                _process_chunk(context, chunk)

            wrapped_iter = immutable_wrap_sync_iterator(
                traced_iter,
                yield_hook=yield_hook,
            )

            class WrappedStream:
                def __init__(self, s: Any) -> None:
                    self._stream = s

                def __iter__(self) -> Iterator[Any]:
                    return wrapped_iter()

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._stream, name)

                def __enter__(self) -> Any:
                    self._stream.__enter__()
                    return self

                def __exit__(self, *args: Any) -> Any:
                    return self._stream.__exit__(*args)

            return WrappedStream(stream)

        def _process_chunk_dict(context: Dict[str, Any], data: Dict[str, Any]) -> None:
            model = data.get("model")
            if model:
                context["model"] = model
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    context["accumulated_content"] = (
                        context.get("accumulated_content", "") + content
                    )
            usage = data.get("usage")
            if usage and isinstance(usage, dict) and usage.get("prompt_tokens"):
                context["usage_dict"] = usage

        def _process_chunk(context: Dict[str, Any], chunk: Any) -> None:
            if hasattr(chunk, "model") and chunk.model:
                context["model"] = chunk.model
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    context["accumulated_content"] = (
                        context.get("accumulated_content", "") + delta.content
                    )
            if hasattr(chunk, "usage") and chunk.usage:
                context["usage"] = chunk.usage

        def _process_completion(context: Dict[str, Any], result: Any) -> None:
            if hasattr(result, "model") and result.model:
                context["model"] = result.model
            if hasattr(result, "choices") and result.choices:
                message = result.choices[0].message
                if message and hasattr(message, "content") and message.content:
                    context["accumulated_content"] = message.content
            if hasattr(result, "usage") and result.usage:
                context["usage"] = result.usage

        def _process_json(context: Dict[str, Any], data: Dict[str, Any]) -> None:
            model = data.get("model")
            if model:
                context["model"] = model
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                content = msg.get("content") if isinstance(msg, dict) else None
                if content:
                    context["accumulated_content"] = content
            usage = data.get("usage")
            if usage and isinstance(usage, dict):
                context["usage_dict"] = usage

        def _finalize_span(
            context: Dict[str, Any], error: BaseException | None
        ) -> None:
            span = context.get("span")
            if not span:
                return

            accumulated = context.get("accumulated_content", "")
            if accumulated:
                span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, accumulated)

            model = context.get("model")
            if model:
                span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, model)

            usage = context.get("usage")
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                total_tokens = usage.total_tokens or 0
                cache_read = 0
                if usage.prompt_tokens_details:
                    cache_read = usage.prompt_tokens_details.cached_tokens or 0

                set_cost_attribute(span, usage)
                prompt_tokens, completion_tokens, cache_read, _ = (
                    openai_tokens_converter(
                        prompt_tokens, completion_tokens, cache_read, 0, total_tokens
                    )
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS, prompt_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage)
                )

            elif context.get("usage_dict"):
                usage_dict = context["usage_dict"]
                prompt_tokens = usage_dict.get("prompt_tokens") or 0
                completion_tokens = usage_dict.get("completion_tokens") or 0
                total_tokens = usage_dict.get("total_tokens") or 0
                cache_read = 0
                prompt_details = usage_dict.get("prompt_tokens_details")
                if prompt_details and isinstance(prompt_details, dict):
                    cache_read = prompt_details.get("cached_tokens") or 0

                prompt_tokens, completion_tokens, cache_read, _ = (
                    openai_tokens_converter(
                        prompt_tokens, completion_tokens, cache_read, 0, total_tokens
                    )
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS, prompt_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage_dict)
                )

            if error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR))

            span.end()

        return WrappedResponseContextManager(original_cm)

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))

    wrapped = mutable_wrap_sync(
        original_func,  # type: ignore[arg-type]
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )

    setattr(client.chat.completions.with_streaming_response, "create", wrapped)


def wrap_with_streaming_response_async(tracer: BaseTracer, client: AsyncOpenAI) -> None:
    original_func = client.chat.completions.with_streaming_response.create

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_hook(ctx: Dict[str, Any], result: Any) -> Any:
        original_cm = result

        class WrappedAsyncResponseContextManager:
            def __init__(self, cm: Any) -> None:
                self._cm = cm

            async def __aenter__(self) -> Any:
                response = await self._cm.__aenter__()
                return _wrap_response(ctx, response)

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
                try:
                    return await self._cm.__aexit__(exc_type, exc_val, exc_tb)
                finally:
                    _finalize_span(ctx, exc_val)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cm, name)

        def _wrap_response(
            context: Dict[str, Any], response: AsyncAPIResponse[Any]
        ) -> Any:
            original_iter_lines = response.iter_lines

            async def traced_iter_lines() -> AsyncGenerator[str, None]:
                async for line in original_iter_lines():
                    yield line

            def yield_hook(inner_ctx: Dict[str, Any], line: str) -> None:
                if not line.startswith("data: ") or line == "data: [DONE]":
                    return
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    return
                if not isinstance(data, dict):
                    return
                _process_chunk_dict(context, data)

            wrapped_iter_lines = immutable_wrap_async_iterator(
                traced_iter_lines,
                yield_hook=yield_hook,
            )

            class WrappedAsyncAPIResponse:
                def __init__(self, resp: AsyncAPIResponse[Any]) -> None:
                    self._response = resp

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._response, name)

                async def iter_lines(self) -> AsyncGenerator[str, None]:
                    async for line in wrapped_iter_lines():
                        yield line

                async def parse(self, *, to: type | None = None) -> Any:
                    result = (
                        await self._response.parse(to=to)
                        if to
                        else await self._response.parse()
                    )
                    if hasattr(result, "__aiter__") and hasattr(result, "response"):
                        return _wrap_stream(context, result)
                    _process_completion(context, result)
                    return result

                async def json(self) -> object:
                    result = await self._response.json()
                    if isinstance(result, dict):
                        _process_json(context, result)
                    return result

                async def read(self) -> bytes:
                    return await self._response.read()

                async def text(self) -> str:
                    return await self._response.text()

                async def iter_text(
                    self, chunk_size: int | None = None
                ) -> AsyncGenerator[str, None]:
                    async for chunk in self._response.iter_text(chunk_size):
                        yield chunk

                async def iter_bytes(
                    self, chunk_size: int | None = None
                ) -> AsyncGenerator[bytes, None]:
                    async for chunk in self._response.iter_bytes(chunk_size):
                        yield chunk

                async def close(self) -> None:
                    return await self._response.close()

            return WrappedAsyncAPIResponse(response)

        def _wrap_stream(context: Dict[str, Any], stream: Any) -> Any:
            original_aiter = stream.__aiter__

            async def traced_aiter() -> AsyncGenerator[Any, None]:
                async for chunk in original_aiter():
                    yield chunk

            def yield_hook(inner_ctx: Dict[str, Any], chunk: Any) -> None:
                _process_chunk(context, chunk)

            wrapped_aiter = immutable_wrap_async_iterator(
                traced_aiter,
                yield_hook=yield_hook,
            )

            class WrappedAsyncStream:
                def __init__(self, s: Any) -> None:
                    self._stream = s

                async def __aiter__(self) -> AsyncGenerator[Any, None]:
                    async for chunk in wrapped_aiter():
                        yield chunk

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._stream, name)

                async def __aenter__(self) -> Any:
                    await self._stream.__aenter__()
                    return self

                async def __aexit__(self, *args: Any) -> Any:
                    return await self._stream.__aexit__(*args)

            return WrappedAsyncStream(stream)

        def _process_chunk_dict(context: Dict[str, Any], data: Dict[str, Any]) -> None:
            model = data.get("model")
            if model:
                context["model"] = model
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    context["accumulated_content"] = (
                        context.get("accumulated_content", "") + content
                    )
            usage = data.get("usage")
            if usage and isinstance(usage, dict) and usage.get("prompt_tokens"):
                context["usage_dict"] = usage

        def _process_chunk(context: Dict[str, Any], chunk: Any) -> None:
            if hasattr(chunk, "model") and chunk.model:
                context["model"] = chunk.model
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    context["accumulated_content"] = (
                        context.get("accumulated_content", "") + delta.content
                    )
            if hasattr(chunk, "usage") and chunk.usage:
                context["usage"] = chunk.usage

        def _process_completion(context: Dict[str, Any], result: Any) -> None:
            if hasattr(result, "model") and result.model:
                context["model"] = result.model
            if hasattr(result, "choices") and result.choices:
                message = result.choices[0].message
                if message and hasattr(message, "content") and message.content:
                    context["accumulated_content"] = message.content
            if hasattr(result, "usage") and result.usage:
                context["usage"] = result.usage

        def _process_json(context: Dict[str, Any], data: Dict[str, Any]) -> None:
            model = data.get("model")
            if model:
                context["model"] = model
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                content = msg.get("content") if isinstance(msg, dict) else None
                if content:
                    context["accumulated_content"] = content
            usage = data.get("usage")
            if usage and isinstance(usage, dict):
                context["usage_dict"] = usage

        def _finalize_span(
            context: Dict[str, Any], error: BaseException | None
        ) -> None:
            span = context.get("span")
            if not span:
                return

            accumulated = context.get("accumulated_content", "")
            if accumulated:
                span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, accumulated)

            model = context.get("model")
            if model:
                span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, model)

            usage = context.get("usage")
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                total_tokens = usage.total_tokens or 0
                cache_read = 0
                if usage.prompt_tokens_details:
                    cache_read = usage.prompt_tokens_details.cached_tokens or 0

                set_cost_attribute(span, usage)
                prompt_tokens, completion_tokens, cache_read, _ = (
                    openai_tokens_converter(
                        prompt_tokens, completion_tokens, cache_read, 0, total_tokens
                    )
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS, prompt_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage)
                )

            elif context.get("usage_dict"):
                usage_dict = context["usage_dict"]
                prompt_tokens = usage_dict.get("prompt_tokens") or 0
                completion_tokens = usage_dict.get("completion_tokens") or 0
                total_tokens = usage_dict.get("total_tokens") or 0
                cache_read = 0
                prompt_details = usage_dict.get("prompt_tokens_details")
                if prompt_details and isinstance(prompt_details, dict):
                    cache_read = prompt_details.get("cached_tokens") or 0

                prompt_tokens, completion_tokens, cache_read, _ = (
                    openai_tokens_converter(
                        prompt_tokens, completion_tokens, cache_read, 0, total_tokens
                    )
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS, prompt_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage_dict)
                )

            if error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR))

            span.end()

        return WrappedAsyncResponseContextManager(original_cm)

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))

    wrapped = mutable_wrap_sync(
        original_func,  # type: ignore[arg-type]
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )

    setattr(client.chat.completions.with_streaming_response, "create", wrapped)
