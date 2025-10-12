from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    AsyncIterator,
    Generator,
    AsyncGenerator,
)

from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    mutable_wrap_sync,
    mutable_wrap_async,
    immutable_wrap_sync_generator,
    immutable_wrap_async_generator,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from openai import OpenAI, AsyncOpenAI
    from openai.types.responses import Response


def wrap_responses_create_sync(tracer: Tracer, client: OpenAI) -> None:
    original_func = client.responses.create

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["is_streaming"] = kwargs.get("stream", False)

        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer.add_agent_attributes_to_span(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )

        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_REQUEST_MODEL, ctx["model_name"]
        )

        if ctx["is_streaming"]:
            ctx["accumulated_content"] = ""

    def post_hook(ctx: Dict[str, Any], result: Any) -> None:
        if not ctx.get("is_streaming", False):
            span = ctx.get("span")
            if not span:
                return

            response: Response = result
            set_span_attribute(
                span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(response)
            )

            usage_data = response.usage
            if usage_data:
                prompt_tokens = usage_data.input_tokens or 0
                completion_tokens = usage_data.output_tokens or 0
                cache_read = 0

                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
                )
                set_span_attribute(
                    span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(usage_data),
                )

            set_span_attribute(
                span,
                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                response.model or ctx["model_name"],
            )

    def mutate_hook(ctx: Dict[str, Any], result: Any) -> Any:
        if not ctx.get("is_streaming", False):
            return result

        stream_iterator: Iterator[Any] = result

        def traced_generator() -> Generator[Any, None, None]:
            for chunk in stream_iterator:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: Any) -> None:
            span = ctx.get("span")
            if not span:
                return

            if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + delta.content
                    )

            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.input_tokens or 0
                completion_tokens = chunk.usage.output_tokens or 0
                cache_read = 0

                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
                )
                set_span_attribute(
                    span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage),
                )

        def post_hook_inner(inner_ctx: Dict[str, Any], result: None) -> None:
            span = ctx.get("span")
            if span:
                accumulated = ctx.get("accumulated_content", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_COMPLETION, accumulated)

        def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
            span = ctx.get("span")
            if span:
                span.record_exception(error)

        def finally_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                span.end()

        wrapped_generator = immutable_wrap_sync_generator(
            traced_generator,
            yield_hook=yield_hook,
            post_hook=post_hook_inner,
            error_hook=error_hook_inner,
            finally_hook=finally_hook_inner,
        )

        return wrapped_generator()

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    def finally_hook(ctx: Dict[str, Any]) -> None:
        if not ctx.get("is_streaming", False):
            span = ctx.get("span")
            if span:
                span.end()

    wrapped = mutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )

    setattr(client.responses, "create", wrapped)


def wrap_responses_create_async(tracer: Tracer, client: AsyncOpenAI) -> None:
    original_func = client.responses.create

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["is_streaming"] = kwargs.get("stream", False)

        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer.add_agent_attributes_to_span(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )

        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_REQUEST_MODEL, ctx["model_name"]
        )

        if ctx["is_streaming"]:
            ctx["accumulated_content"] = ""

    def post_hook(ctx: Dict[str, Any], result: Any) -> None:
        if not ctx.get("is_streaming", False):
            span = ctx.get("span")
            if not span:
                return

            response: Response = result
            set_span_attribute(
                span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(response)
            )

            usage_data = response.usage
            if usage_data:
                prompt_tokens = usage_data.input_tokens or 0
                completion_tokens = usage_data.output_tokens or 0
                cache_read = 0

                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
                )
                set_span_attribute(
                    span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(usage_data),
                )

            set_span_attribute(
                span,
                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                response.model or ctx["model_name"],
            )

    def mutate_hook(ctx: Dict[str, Any], result: Any) -> Any:
        if not ctx.get("is_streaming", False):
            return result

        stream_iterator: AsyncIterator[Any] = result

        async def traced_generator() -> AsyncGenerator[Any, None]:
            async for chunk in stream_iterator:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: Any) -> None:
            span = ctx.get("span")
            if not span:
                return

            if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + delta.content
                    )

            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.input_tokens or 0
                completion_tokens = chunk.usage.output_tokens or 0
                cache_read = 0

                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
                )
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
                )
                set_span_attribute(
                    span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage),
                )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                accumulated = ctx.get("accumulated_content", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_COMPLETION, accumulated)

        def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
            span = ctx.get("span")
            if span:
                span.record_exception(error)

        def finally_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                span.end()

        wrapped_generator = immutable_wrap_async_generator(
            traced_generator,
            yield_hook=yield_hook,
            post_hook=post_hook_inner,
            error_hook=error_hook_inner,
            finally_hook=finally_hook_inner,
        )

        return wrapped_generator()

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    def finally_hook(ctx: Dict[str, Any]) -> None:
        if not ctx.get("is_streaming", False):
            span = ctx.get("span")
            if span:
                span.end()

    wrapped = mutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )

    setattr(client.responses, "create", wrapped)
