from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    AsyncIterator,
    Generator,
    AsyncGenerator,
    Union,
)

from opentelemetry.trace import Span, Status, StatusCode
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    immutable_wrap_sync,
    immutable_wrap_async,
    mutable_wrap_sync,
    mutable_wrap_async,
    immutable_wrap_sync_iterator,
    immutable_wrap_async_iterator,
)
from judgeval.instrumentation.llm.llm_openai.utils import (
    openai_tokens_converter,
    set_cost_attribute,
)

from judgeval.trace import BaseTracer

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.image_edit_completed_event import Usage as ImageEditUsage
    from openai.types.image_gen_completed_event import Usage as ImageGenUsage
    from openai.types.image_edit_stream_event import ImageEditStreamEvent
    from openai.types.image_gen_stream_event import ImageGenStreamEvent
    from openai.types.images_response import ImagesResponse, Usage

    ImageStreamEvent = Union[ImageGenStreamEvent, ImageEditStreamEvent]
    ImageUsage = Union[Usage, ImageGenUsage, ImageEditUsage]

_IMAGE_COMPLETED_TYPES = frozenset(
    {"image_generation.completed", "image_edit.completed"}
)


def _set_image_usage(span: Span, usage_data: ImageUsage) -> None:
    input_text_tokens = usage_data.input_tokens_details.text_tokens or 0
    input_image_tokens = usage_data.input_tokens_details.image_tokens or 0
    output_tokens = usage_data.output_tokens or 0

    set_cost_attribute(span, usage_data)
    (
        input_text_tokens,
        _,
        _,
        _,
        input_image_tokens,
        output_image_tokens,
    ) = openai_tokens_converter(
        input_text_tokens,
        0,
        0,
        0,
        input_image_tokens,
        output_tokens,
        usage_data.total_tokens,
    )

    span.set_attribute(
        AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS, input_text_tokens
    )
    span.set_attribute(
        AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_IMAGE_TOKENS, input_image_tokens
    )
    span.set_attribute(
        AttributeKeys.JUDGMENT_USAGE_OUTPUT_IMAGE_TOKENS, output_image_tokens
    )
    span.set_attribute(
        AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage_data)
    )


def _wrap_images_non_streaming_sync(
    original_func: Callable[..., ImagesResponse],
) -> Callable[..., ImagesResponse]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "OPENAI_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: ImagesResponse) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(
            AttributeKeys.GEN_AI_COMPLETION,
            safe_serialize(result),
        )

        usage_data = result.usage
        if usage_data:
            _set_image_usage(span, usage_data)

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))

    def finally_hook(ctx: Dict[str, Any]) -> None:
        span = ctx.get("span")
        if span:
            span.end()

    return immutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )


def _wrap_images_non_streaming_async(
    original_func: Callable[..., Awaitable[ImagesResponse]],
) -> Callable[..., Awaitable[ImagesResponse]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "OPENAI_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: ImagesResponse) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(
            AttributeKeys.GEN_AI_COMPLETION,
            safe_serialize(result),
        )

        usage_data = result.usage
        if usage_data:
            _set_image_usage(span, usage_data)

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))

    def finally_hook(ctx: Dict[str, Any]) -> None:
        span = ctx.get("span")
        if span:
            span.end()

    return immutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )


def _wrap_images_streaming_sync(
    original_func: Callable[..., Iterator[ImageStreamEvent]],
) -> Callable[..., Iterator[ImageStreamEvent]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "OPENAI_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def mutate_hook(
        ctx: Dict[str, Any], result: Iterator[ImageStreamEvent]
    ) -> Iterator[ImageStreamEvent]:
        def traced_generator() -> Generator[ImageStreamEvent, None, None]:
            for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: ImageStreamEvent) -> None:
            span = ctx.get("span")
            if not span:
                return

            if chunk.type in _IMAGE_COMPLETED_TYPES:
                ctx["completion_data"] = chunk

                if hasattr(chunk, "usage") and chunk.usage:
                    _set_image_usage(span, chunk.usage)

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                completion = ctx.get("completion_data", {})
                span.set_attribute(
                    AttributeKeys.GEN_AI_COMPLETION, safe_serialize(completion)
                )

        def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
            span = ctx.get("span")
            if span:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR))

        def finally_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                span.end()

        wrapped_generator = immutable_wrap_sync_iterator(
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
            span.set_status(Status(StatusCode.ERROR))

    return mutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )


def _wrap_images_streaming_async(
    original_func: Callable[..., Awaitable[AsyncIterator[ImageStreamEvent]]],
) -> Callable[..., Awaitable[AsyncIterator[ImageStreamEvent]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "OPENAI_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def mutate_hook(
        ctx: Dict[str, Any], result: AsyncIterator[ImageStreamEvent]
    ) -> AsyncIterator[ImageStreamEvent]:
        async def traced_generator() -> AsyncGenerator[ImageStreamEvent, None]:
            async for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: ImageStreamEvent) -> None:
            span = ctx.get("span")
            if not span:
                return

            if chunk.type in _IMAGE_COMPLETED_TYPES:
                ctx["completion_data"] = chunk

                if hasattr(chunk, "usage") and chunk.usage:
                    _set_image_usage(span, chunk.usage)

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                completion = ctx.get("completion_data", {})
                span.set_attribute(
                    AttributeKeys.GEN_AI_COMPLETION, safe_serialize(completion)
                )

        def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
            span = ctx.get("span")
            if span:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR))

        def finally_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                span.end()

        wrapped_generator = immutable_wrap_async_iterator(
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
            span.set_status(Status(StatusCode.ERROR))

    return mutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )


def wrap_images_generate_sync(client: OpenAI) -> None:
    original_func = client.images.generate

    def dispatcher(*args: Any, **kwargs: Any) -> Any:
        extra_headers = kwargs.get("extra_headers") or {}
        if (
            isinstance(extra_headers, dict)
            and extra_headers.get("X-Stainless-Raw-Response") == "stream"
        ):
            return original_func(*args, **kwargs)

        if kwargs.get("stream", False):
            return _wrap_images_streaming_sync(original_func)(*args, **kwargs)
        return _wrap_images_non_streaming_sync(original_func)(*args, **kwargs)

    setattr(client.images, "generate", dispatcher)


def wrap_images_generate_async(client: AsyncOpenAI) -> None:
    original_func = client.images.generate

    async def dispatcher(*args: Any, **kwargs: Any) -> Any:
        extra_headers = kwargs.get("extra_headers") or {}
        if (
            isinstance(extra_headers, dict)
            and extra_headers.get("X-Stainless-Raw-Response") == "stream"
        ):
            return await original_func(*args, **kwargs)

        if kwargs.get("stream", False):
            return await _wrap_images_streaming_async(original_func)(*args, **kwargs)
        return await _wrap_images_non_streaming_async(original_func)(*args, **kwargs)

    setattr(client.images, "generate", dispatcher)
