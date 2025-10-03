from __future__ import annotations
import functools
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Protocol,
    Tuple,
    Union,
    Iterator,
    AsyncIterator,
    Sequence,
    runtime_checkable,
)

from judgeval.tracer.llm.llm_google.config import (
    google_genai_Client,
    google_genai_AsyncClient,
)
from judgeval.tracer.managers import sync_span_context, async_span_context
from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from opentelemetry.trace import Span

# Keep the original client type for runtime compatibility
GoogleClientType = Union[google_genai_Client, google_genai_AsyncClient]


# Usage protocols
@runtime_checkable
class GoogleUsageMetadata(Protocol):
    prompt_token_count: Optional[int]
    candidates_token_count: Optional[int]
    total_token_count: Optional[int]
    cached_content_token_count: Optional[int]


# Content protocols
@runtime_checkable
class GooglePart(Protocol):
    text: str


@runtime_checkable
class GoogleContent(Protocol):
    parts: Sequence[GooglePart]


@runtime_checkable
class GoogleCandidate(Protocol):
    content: GoogleContent
    finish_reason: Optional[str]


@runtime_checkable
class GoogleGenerateContentResponse(Protocol):
    candidates: Sequence[GoogleCandidate]
    usage_metadata: Optional[GoogleUsageMetadata]
    model_version: Optional[str]


# Stream protocols
@runtime_checkable
class GoogleStreamChunk(Protocol):
    candidates: Sequence[GoogleCandidate]
    usage_metadata: Optional[GoogleUsageMetadata]


# Client protocols
@runtime_checkable
class GoogleClient(Protocol):
    pass


@runtime_checkable
class GoogleAsyncClient(Protocol):
    pass


# Union types
GoogleResponseType = GoogleGenerateContentResponse
GoogleStreamType = Union[Iterator[GoogleStreamChunk], AsyncIterator[GoogleStreamChunk]]


def _extract_google_content(chunk: GoogleStreamChunk) -> str:
    if chunk.candidates and len(chunk.candidates) > 0:
        candidate = chunk.candidates[0]
        if (
            candidate.content
            and candidate.content.parts
            and len(candidate.content.parts) > 0
        ):
            return candidate.content.parts[0].text or ""
    return ""


def _extract_google_tokens(
    usage_data: GoogleUsageMetadata,
) -> Tuple[int, int, int, int]:
    prompt_tokens = usage_data.prompt_token_count or 0
    completion_tokens = usage_data.candidates_token_count or 0
    cache_read_input_tokens = usage_data.cached_content_token_count or 0
    cache_creation_input_tokens = 0  # Google GenAI doesn't have cache creation tokens
    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def _format_google_output(
    response: GoogleGenerateContentResponse,
) -> Tuple[Optional[str], Optional[GoogleUsageMetadata]]:
    message_content: Optional[str] = None
    usage_data: Optional[GoogleUsageMetadata] = None

    try:
        if isinstance(response, GoogleGenerateContentResponse):
            usage_data = response.usage_metadata
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if (
                    candidate.content
                    and candidate.content.parts
                    and len(candidate.content.parts) > 0
                ):
                    message_content = candidate.content.parts[0].text
    except (AttributeError, IndexError, TypeError):
        pass

    return message_content, usage_data


class TracedGoogleGenerator:
    def __init__(
        self,
        tracer: Tracer,
        generator: Iterator[GoogleStreamChunk],
        client: GoogleClientType,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.generator = generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __iter__(self) -> Iterator[GoogleStreamChunk]:
        return self

    def __next__(self) -> GoogleStreamChunk:
        try:
            chunk = next(self.generator)
            content = _extract_google_content(chunk)
            if content:
                self.accumulated_content += content
            if chunk.usage_metadata:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_google_tokens(chunk.usage_metadata)
                )
                set_span_attribute(
                    self.span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                    completion_tokens,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                    cache_read,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage_metadata),
                )
            return chunk
        except StopIteration:
            set_span_attribute(
                self.span, AttributeKeys.GEN_AI_COMPLETION, self.accumulated_content
            )
            self.span.end()
            raise
        except Exception as e:
            if self.span:
                self.span.record_exception(e)
                self.span.end()
            raise


class TracedGoogleAsyncGenerator:
    def __init__(
        self,
        tracer: Tracer,
        async_generator: AsyncIterator[GoogleStreamChunk],
        client: GoogleClientType,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.async_generator = async_generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __aiter__(self) -> AsyncIterator[GoogleStreamChunk]:
        return self

    async def __anext__(self) -> GoogleStreamChunk:
        try:
            chunk = await self.async_generator.__anext__()
            content = _extract_google_content(chunk)
            if content:
                self.accumulated_content += content
            if chunk.usage_metadata:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_google_tokens(chunk.usage_metadata)
                )
                set_span_attribute(
                    self.span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                    completion_tokens,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                    cache_read,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage_metadata),
                )
            return chunk
        except StopAsyncIteration:
            set_span_attribute(
                self.span, AttributeKeys.GEN_AI_COMPLETION, self.accumulated_content
            )
            self.span.end()
            raise
        except Exception as e:
            if self.span:
                self.span.record_exception(e)
                self.span.end()
            raise


def wrap_google_client(tracer: Tracer, client: GoogleClientType) -> GoogleClientType:
    def wrapped(function: Callable, span_name: str):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if kwargs.get("stream", False):
                span = tracer.get_tracer().start_span(
                    span_name, attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                )
                tracer.add_agent_attributes_to_span(span)
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                )
                model_name = kwargs.get("model", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name)
                stream_response = function(*args, **kwargs)
                return TracedGoogleGenerator(
                    tracer, stream_response, client, span, model_name
                )
            else:
                with sync_span_context(
                    tracer, span_name, {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                ) as span:
                    tracer.add_agent_attributes_to_span(span)
                    set_span_attribute(
                        span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                    )
                    model_name = kwargs.get("model", "")
                    set_span_attribute(
                        span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name
                    )
                    response = function(*args, **kwargs)

                    if isinstance(response, GoogleGenerateContentResponse):
                        output, usage_data = _format_google_output(response)
                        set_span_attribute(
                            span, AttributeKeys.GEN_AI_COMPLETION, output
                        )
                        if usage_data:
                            (
                                prompt_tokens,
                                completion_tokens,
                                cache_read,
                                cache_creation,
                            ) = _extract_google_tokens(usage_data)
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS,
                                prompt_tokens,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                                completion_tokens,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                                cache_read,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.JUDGMENT_USAGE_METADATA,
                                safe_serialize(usage_data),
                            )
                        set_span_attribute(
                            span,
                            AttributeKeys.GEN_AI_RESPONSE_MODEL,
                            getattr(response, "model_version", model_name),
                        )
                    return response

        return wrapper

    def wrapped_async(function: Callable, span_name: str):
        @functools.wraps(function)
        async def wrapper(*args, **kwargs):
            if kwargs.get("stream", False):
                span = tracer.get_tracer().start_span(
                    span_name, attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                )
                tracer.add_agent_attributes_to_span(span)
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                )
                model_name = kwargs.get("model", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name)
                stream_response = await function(*args, **kwargs)
                return TracedGoogleAsyncGenerator(
                    tracer, stream_response, client, span, model_name
                )
            else:
                async with async_span_context(
                    tracer, span_name, {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                ) as span:
                    tracer.add_agent_attributes_to_span(span)
                    set_span_attribute(
                        span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                    )
                    model_name = kwargs.get("model", "")
                    set_span_attribute(
                        span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name
                    )
                    response = await function(*args, **kwargs)

                    if isinstance(response, GoogleGenerateContentResponse):
                        output, usage_data = _format_google_output(response)
                        set_span_attribute(
                            span, AttributeKeys.GEN_AI_COMPLETION, output
                        )
                        if usage_data:
                            (
                                prompt_tokens,
                                completion_tokens,
                                cache_read,
                                cache_creation,
                            ) = _extract_google_tokens(usage_data)
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS,
                                prompt_tokens,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                                completion_tokens,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                                cache_read,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.JUDGMENT_USAGE_METADATA,
                                safe_serialize(usage_data),
                            )
                        set_span_attribute(
                            span,
                            AttributeKeys.GEN_AI_RESPONSE_MODEL,
                            getattr(response, "model_version", model_name),
                        )
                    return response

        return wrapper

    span_name = "GOOGLE_API_CALL"
    if google_genai_Client is not None and isinstance(client, google_genai_Client):
        # Type narrowing for mypy
        google_client = client  # type: ignore[assignment]
        setattr(
            google_client.models,
            "generate_content",
            wrapped(google_client.models.generate_content, span_name),
        )
    elif google_genai_AsyncClient is not None and isinstance(
        client, google_genai_AsyncClient
    ):
        # Type narrowing for mypy
        async_google_client = client  # type: ignore[assignment]
        setattr(
            async_google_client.models,
            "generate_content",
            wrapped_async(async_google_client.models.generate_content, span_name),
        )

    return client
