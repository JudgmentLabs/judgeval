from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
)

from opentelemetry.trace import Status, StatusCode
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    immutable_wrap_sync,
    immutable_wrap_async,
)
from judgeval.v1.instrumentation.llm.llm_openai.utils import set_cost_attribute

if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer
    from openai import OpenAI, AsyncOpenAI
    from openai.types.images_response import ImagesResponse


def _make_pre_hook(
    tracer: BaseTracer,
) -> Callable[..., None]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_PROMPT, safe_serialize(kwargs)
        )
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    return pre_hook


def _images_post_hook(ctx: Dict[str, Any], result: ImagesResponse) -> None:
    span = ctx.get("span")
    if not span:
        return

    span.set_attribute(AttributeKeys.JUDGMENT_LLM_COMPLETION, safe_serialize(result))

    usage_data = getattr(result, "usage", None)
    if usage_data:
        input_tokens = getattr(usage_data, "input_tokens", 0) or 0
        output_tokens = getattr(usage_data, "output_tokens", 0) or 0

        cache_read = 0
        input_details = getattr(usage_data, "input_tokens_details", None)
        if input_details:
            cache_read = getattr(input_details, "cached_tokens", 0) or 0

        set_cost_attribute(span, usage_data)

        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
            max(input_tokens - cache_read, 0),
        )
        span.set_attribute(AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, output_tokens)
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
        )
        span.set_attribute(AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0)
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_METADATA,
            safe_serialize(usage_data),
        )


def _error_hook(ctx: Dict[str, Any], error: Exception) -> None:
    span = ctx.get("span")
    if span:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR))


def _finally_hook(ctx: Dict[str, Any]) -> None:
    span = ctx.get("span")
    if span:
        span.end()


def _wrap_images_sync(
    tracer: BaseTracer, original_func: Callable[..., ImagesResponse]
) -> Callable[..., ImagesResponse]:
    return immutable_wrap_sync(
        original_func,
        pre_hook=_make_pre_hook(tracer),
        post_hook=_images_post_hook,
        error_hook=_error_hook,
        finally_hook=_finally_hook,
    )


def _wrap_images_async(
    tracer: BaseTracer, original_func: Callable[..., Awaitable[ImagesResponse]]
) -> Callable[..., Awaitable[ImagesResponse]]:
    return immutable_wrap_async(
        original_func,
        pre_hook=_make_pre_hook(tracer),
        post_hook=_images_post_hook,
        error_hook=_error_hook,
        finally_hook=_finally_hook,
    )


def wrap_images_generate_sync(tracer: BaseTracer, client: OpenAI) -> None:
    setattr(
        client.images, "generate", _wrap_images_sync(tracer, client.images.generate)
    )


def wrap_images_edit_sync(tracer: BaseTracer, client: OpenAI) -> None:
    setattr(client.images, "edit", _wrap_images_sync(tracer, client.images.edit))


def wrap_images_create_variation_sync(tracer: BaseTracer, client: OpenAI) -> None:
    setattr(
        client.images,
        "create_variation",
        _wrap_images_sync(tracer, client.images.create_variation),
    )


def wrap_images_generate_async(tracer: BaseTracer, client: AsyncOpenAI) -> None:
    setattr(
        client.images, "generate", _wrap_images_async(tracer, client.images.generate)
    )


def wrap_images_edit_async(tracer: BaseTracer, client: AsyncOpenAI) -> None:
    setattr(client.images, "edit", _wrap_images_async(tracer, client.images.edit))


def wrap_images_create_variation_async(tracer: BaseTracer, client: AsyncOpenAI) -> None:
    setattr(
        client.images,
        "create_variation",
        _wrap_images_async(tracer, client.images.create_variation),
    )
