from __future__ import annotations

import contextvars
import functools
import inspect
import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    TypedDict,
    TypeVar,
    cast,
    overload,
)
from uuid import uuid4
from opentelemetry.trace import Span, Status, StatusCode, Tracer as OTELTracer
from opentelemetry.sdk.trace import TracerProvider

from judgeval.judgment_attribute_keys import AttributeKeys, InternalAttributeKeys
from judgeval.utils.decorators.debug_time import debug_time
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.utils.serialize import serialize_attribute, safe_serialize
from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
import judgeval.trace.baggage as baggage
from judgeval.trace.generators import (
    _ObservedSyncGenerator,
    _ObservedAsyncGenerator,
)
from judgeval.background_queue import enqueue as bg_enqueue

if TYPE_CHECKING:
    from judgeval.internal.api import JudgmentSyncClient
    from judgeval.trace.processors.judgment_span_processor import (
        JudgmentSpanProcessor,
    )
    from judgeval.trace.exporters.judgment_span_exporter import JudgmentSpanExporter

TClient = TypeVar("TClient")
C = TypeVar("C", bound=Callable[..., Any])


class LLMMetadata(TypedDict, total=False):
    """Token usage and cost metadata for an LLM call.

    Pass to `Tracer.recordLLMMetadata()` to attach cost and token
    breakdown to the current span. All fields are optional.

    Attributes:
        model: Model identifier, e.g. `"gpt-4o"` or `"claude-sonnet-4-20250514"`.
        provider: Provider name, e.g. `"openai"` or `"anthropic"`.
        non_cached_input_tokens: Input tokens that were not served from cache.
        output_tokens: Tokens generated in the response.
        cache_read_input_tokens: Input tokens served from prompt cache.
        cache_creation_input_tokens: Input tokens used to create a new cache entry.
        total_cost_usd: Total cost of the call in USD.

    Examples:
        ```python
        Tracer.recordLLMMetadata({
            "model": "gpt-4o",
            "provider": "openai",
            "non_cached_input_tokens": 150,
            "output_tokens": 80,
            "total_cost_usd": 0.003,
        })
        ```
    """

    model: str
    provider: str
    non_cached_input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    total_cost_usd: float


class BaseTracer(ABC):
    """Base class providing the tracing API surface.

    You don't instantiate this directly -- use `Tracer.init()` instead. The
    static methods below (`observe`, `wrap`, `set_attribute`, `span`, etc.)
    are the primary API you'll use after initializing a tracer.
    """

    __slots__ = (
        "project_name",
        "project_id",
        "api_key",
        "organization_id",
        "api_url",
        "environment",
        "serializer",
        "_tracer_provider",
        "_client",
    )

    # ------------------------------------------------------------------ #
    #  Initialization                                                     #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        project_name: Optional[str],
        project_id: Optional[str],
        api_key: Optional[str],
        organization_id: Optional[str],
        api_url: Optional[str],
        environment: Optional[str],
        serializer: Callable[[Any], str],
        tracer_provider: TracerProvider,
        client: Optional[JudgmentSyncClient],
    ):
        self.project_name = project_name
        self.project_id = project_id
        self.api_key = api_key
        self.organization_id = organization_id
        self.api_url = api_url
        self.environment = environment
        self.serializer = serializer
        self._tracer_provider = tracer_provider
        self._client = client

    # ------------------------------------------------------------------ #
    #  Abstract Lifecycle                                                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_span_processor(self) -> JudgmentSpanProcessor:
        """Return the span processor for this tracer."""

    @abstractmethod
    def get_span_exporter(self) -> JudgmentSpanExporter:
        """Return the span exporter for this tracer."""

    # ------------------------------------------------------------------ #
    #  Internal Helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_proxy_provider() -> JudgmentTracerProvider:
        return JudgmentTracerProvider.get_instance()

    @staticmethod
    def _get_serializer() -> Callable[[Any], str]:
        tracer = BaseTracer._get_proxy_provider().get_active_tracer()
        return tracer.serializer if tracer else safe_serialize

    @staticmethod
    def _get_current_trace_and_span_id() -> Optional[tuple[str, str]]:
        """Return ``(trace_id, span_id)`` as hex strings, or ``None``
        if no valid sampled span is active."""
        proxy = BaseTracer._get_proxy_provider()
        current_span = proxy.get_current_span()
        if current_span is None or not current_span.is_recording():
            return None
        ctx = current_span.get_span_context()
        if not ctx.is_valid or not ctx.trace_flags.sampled:
            return None
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")

    @staticmethod
    @dont_throw
    def _emit_partial() -> None:
        """Ask the active tracer's span processor to emit the current span
        as a partial update without ending it."""
        tracer = BaseTracer._get_proxy_provider().get_active_tracer()
        if tracer is None:
            return
        tracer.get_span_processor().emit_partial()

    # ------------------------------------------------------------------ #
    #  Static API: Span Access & Lifecycle                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_current_span() -> Span:
        """Return the currently active span from the Judgment tracer provider.

        Returns:
            The active ``Span`` object.
        """
        proxy = BaseTracer._get_proxy_provider()
        return proxy.get_current_span()

    @staticmethod
    def force_flush(timeout_millis: int = 30000) -> bool:
        """Send all pending spans to Judgment immediately.

        Call this before your process exits (e.g. in a serverless function)
        to ensure no spans are lost. Does not shut down the tracer.

        Args:
            timeout_millis: Maximum wait time in milliseconds.

        Returns:
            True if all spans were flushed within the timeout.

        Examples:
            ```python
            def lambda_handler(event, context):
                result = process(event)
                Tracer.force_flush()
                return result
            ```
        """
        proxy = BaseTracer._get_proxy_provider()
        return proxy.force_flush(timeout_millis)

    @staticmethod
    def shutdown(timeout_millis: int = 30000) -> None:
        """Flush pending spans and shut down the tracer.

        Call this on application exit to ensure all data is exported
        before the process terminates.

        Args:
            timeout_millis: Maximum wait time in milliseconds.
        """
        proxy = BaseTracer._get_proxy_provider()
        proxy.shutdown()

    @staticmethod
    @dont_throw
    def registerOTELInstrumentation(instrumentor) -> None:
        """Register a third-party OpenTelemetry instrumentor with Judgment.

        Use this to route spans from libraries like `opentelemetry-instrumentation-requests`
        through the Judgment trace pipeline.
        """
        proxy = BaseTracer._get_proxy_provider()
        proxy.add_instrumentation(instrumentor)

    # ------------------------------------------------------------------ #
    #  Static: Span Creation                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_otel_tracer() -> OTELTracer:
        proxy = BaseTracer._get_proxy_provider()
        return proxy.get_tracer(JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME)

    @staticmethod
    def start_span(
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span that must be ended manually with ``span.end()``.

        Prefer the ``span`` context manager for automatic lifecycle management.

        Args:
            name: Name for the new span.
            attributes: Optional dictionary of initial span attributes.

        Returns:
            The newly started ``Span``.
        """
        span = BaseTracer._get_otel_tracer().start_span(name, attributes=attributes)
        BaseTracer._emit_partial()
        return span

    @staticmethod
    @contextmanager
    def start_as_current_span(
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """Start a span and set it as the current span in the context.

        Args:
            name: Name for the new span.
            attributes: Optional dictionary of initial span attributes.

        Yields:
            The newly started ``Span``.
        """
        with BaseTracer._get_otel_tracer().start_as_current_span(
            name, attributes=attributes
        ) as span:
            BaseTracer._emit_partial()
            yield span

    @staticmethod
    @contextmanager
    def span(span_name: str) -> Iterator[Span]:
        """Open a child span using a `with` block.

        Use this for tracing a section of code that isn't a standalone
        function. Exceptions are automatically recorded on the span.

        Args:
            span_name: Name for this span (visible in the dashboard).

        Yields:
            The active `Span` object.

        Examples:
            ```python
            with Tracer.span("process-results"):
                results = parse(raw_data)
                Tracer.set_attribute("result_count", len(results))
            ```
        """
        with BaseTracer.start_as_current_span(span_name) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    # ------------------------------------------------------------------ #
    #  Static API: Observation Decorator                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    @overload
    def observe(
        func: C,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
        disable_generator_yield_span: bool = False,
    ) -> C: ...

    @staticmethod
    @overload
    def observe(
        func: None = None,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
        disable_generator_yield_span: bool = False,
    ) -> Callable[[C], C]: ...

    @staticmethod
    def observe(
        func: Optional[C] = None,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
        disable_generator_yield_span: bool = False,
    ) -> C | Callable[[C], C]:
        """Decorator that automatically traces a function call.

        Wraps any sync or async function in a span. Inputs and outputs are
        captured automatically. Works with or without parentheses.

        Args:
            func: The function to wrap (set implicitly when used as
                `@Tracer.observe` without parentheses).
            span_type: The kind of span. Use `"tool"`, `"agent"`, `"llm"`,
                or `"function"` to categorize work in the dashboard.
                Defaults to `"span"`.
            span_name: Override the span name (defaults to the function name).
            record_input: Capture and store function arguments. Set to False
                for functions with sensitive or very large inputs.
            record_output: Capture and store the return value.
            disable_generator_yield_span: Suppress per-yield child spans for
                generator functions.

        Examples:
            Basic usage:

            ```python
            @Tracer.observe(span_type="tool")
            def search(query: str) -> list[str]:
                return vector_db.search(query)
            ```

            Async functions work the same way:

            ```python
            @Tracer.observe(span_type="agent")
            async def answer(question: str) -> str:
                context = search(question)
                return await llm.generate(question, context)
            ```

            Without parentheses (uses default settings):

            ```python
            @Tracer.observe
            def my_function():
                ...
            ```
        """

        def decorator(f: C) -> C:
            proxy = BaseTracer._get_proxy_provider()
            name = span_name or f.__name__

            if inspect.iscoroutinefunction(f):

                @functools.wraps(f)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    otel_tracer = proxy.get_tracer(
                        JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
                    )
                    with otel_tracer.start_as_current_span(name) as span:
                        if span_type:
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_SPAN_KIND, span_type
                            )
                        try:
                            if record_input:
                                span.set_attribute(
                                    AttributeKeys.JUDGMENT_INPUT,
                                    serialize_attribute(
                                        _format_inputs(f, args, kwargs),
                                        BaseTracer._get_serializer(),
                                    ),
                                )
                            BaseTracer._emit_partial()
                            result = await f(*args, **kwargs)
                            if record_output:
                                span.set_attribute(
                                    AttributeKeys.JUDGMENT_OUTPUT,
                                    serialize_attribute(
                                        result, BaseTracer._get_serializer()
                                    ),
                                )
                            return result
                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise

                return cast(C, async_wrapper)
            else:

                @functools.wraps(f)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    otel_tracer = proxy.get_tracer(
                        JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
                    )
                    with otel_tracer.start_as_current_span(
                        name, end_on_exit=False
                    ) as span:
                        if span_type:
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_SPAN_KIND, span_type
                            )
                        try:
                            if record_input:
                                span.set_attribute(
                                    AttributeKeys.JUDGMENT_INPUT,
                                    serialize_attribute(
                                        _format_inputs(f, args, kwargs),
                                        BaseTracer._get_serializer(),
                                    ),
                                )
                            BaseTracer._emit_partial()
                            result = f(*args, **kwargs)
                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.end()
                            raise

                        serializer = BaseTracer._get_serializer()

                        if inspect.isgenerator(result):
                            if record_output:
                                span.set_attribute(
                                    AttributeKeys.JUDGMENT_OUTPUT, "<generator>"
                                )
                            return _ObservedSyncGenerator(
                                result,
                                span,
                                serializer,
                                otel_tracer,
                                contextvars.copy_context(),
                                disable_generator_yield_span or not record_output,
                            )
                        if inspect.isasyncgen(result):
                            if record_output:
                                span.set_attribute(
                                    AttributeKeys.JUDGMENT_OUTPUT,
                                    "<async_generator>",
                                )
                            return _ObservedAsyncGenerator(
                                result,
                                span,
                                serializer,
                                otel_tracer,
                                contextvars.copy_context(),
                                disable_generator_yield_span or not record_output,
                            )

                        if record_output:
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_OUTPUT,
                                serialize_attribute(result, serializer),
                            )
                        span.end()
                        return result

                return cast(C, sync_wrapper)

        if func is None:
            return decorator
        return decorator(func)

    @staticmethod
    def wrap(client: TClient) -> TClient:
        """Wrap an LLM client for automatic tracing of all API calls.

        Supported providers: **OpenAI**, **Anthropic**, **Together AI**, and
        **Google GenAI**. Once wrapped, every API call made through the client
        is recorded as a span with model name, token counts, and cost.

        Args:
            client: An LLM provider client instance (e.g. `OpenAI()`,
                `Anthropic()`).

        Returns:
            The same client instance, now instrumented with tracing.

        Examples:
            ```python
            from openai import OpenAI
            from anthropic import Anthropic

            openai = Tracer.wrap(OpenAI())
            anthropic = Tracer.wrap(Anthropic())
            ```
        """
        from judgeval.instrumentation.llm import wrap_provider

        return wrap_provider(client)

    # ------------------------------------------------------------------ #
    #  Static: Span Kind                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    @dont_throw
    def set_span_kind(kind: str) -> None:
        """Set the ``judgment.span_kind`` attribute on the current span."""
        if kind is None:
            return
        current_span = BaseTracer._get_proxy_provider().get_current_span()
        if current_span is not None and current_span.is_recording():
            current_span.set_attribute(AttributeKeys.JUDGMENT_SPAN_KIND, kind)

    @staticmethod
    def set_llm_span() -> None:
        BaseTracer.set_span_kind("llm")

    @staticmethod
    def set_tool_span() -> None:
        BaseTracer.set_span_kind("tool")

    @staticmethod
    def set_general_span() -> None:
        BaseTracer.set_span_kind("span")

    # ------------------------------------------------------------------ #
    #  Static: Span Attribute Operations                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    @dont_throw
    def set_attribute(key: str, value: Any) -> None:
        """Attach a custom key-value pair to the current span.

        Use this to record application-specific metadata that you want
        to see in the Judgment dashboard. Non-primitive values (dicts,
        lists, objects) are serialized to strings automatically.

        Args:
            key: Attribute name (e.g. `"user_tier"`, `"search_results_count"`).
            value: The value to record.

        Examples:
            ```python
            Tracer.set_attribute("user_tier", "premium")
            Tracer.set_attribute("search_results_count", len(results))
            ```
        """
        current_span = BaseTracer._get_proxy_provider().get_current_span()
        if current_span is None or not current_span.is_recording():
            return
        if not key or value is None:
            return
        current_span.set_attribute(
            key,
            serialize_attribute(value, BaseTracer._get_serializer()),
        )

    @staticmethod
    def set_attributes(attributes: Dict[str, Any]) -> None:
        """Set multiple custom attributes on the current span at once.

        Args:
            attributes: Dictionary of key-value pairs to set.
        """
        if attributes is None:
            return
        for key, value in attributes.items():
            BaseTracer.set_attribute(key, value)

    @staticmethod
    def set_input(input_data: Any) -> None:
        """Manually set the input for the current span.

        Use when `@observe(record_input=False)` is set but you want to
        record a sanitized or transformed version of the input.

        Args:
            input_data: The input value to record.
        """
        BaseTracer.set_attribute(AttributeKeys.JUDGMENT_INPUT, input_data)

    @staticmethod
    def set_output(output_data: Any) -> None:
        """Manually set the output for the current span.

        Use when `@observe(record_output=False)` is set but you want to
        record a sanitized or transformed version of the output.

        Args:
            output_data: The output value to record.
        """
        BaseTracer.set_attribute(AttributeKeys.JUDGMENT_OUTPUT, output_data)

    @staticmethod
    @dont_throw
    def recordLLMMetadata(metadata: LLMMetadata) -> None:
        """Record model, token usage, and cost on the current span.

        If you're using `Tracer.wrap()` this is called automatically. Use
        this method when you need to record metadata for a custom LLM
        integration.

        Args:
            metadata: A dict with keys like `model`, `provider`,
                `non_cached_input_tokens`, `output_tokens`, and
                `total_cost_usd`. All fields are optional.

        Examples:
            ```python
            @Tracer.observe(span_type="llm")
            def call_custom_model(prompt: str) -> str:
                response = my_model.generate(prompt)
                Tracer.recordLLMMetadata({
                    "model": "my-model-v2",
                    "output_tokens": response.usage.output,
                    "total_cost_usd": response.usage.cost,
                })
                return response.text
            ```
        """
        current_span = BaseTracer._get_proxy_provider().get_current_span()
        if current_span is None or not current_span.is_recording():
            return

        if "model" in metadata:
            current_span.set_attribute(
                AttributeKeys.JUDGMENT_LLM_MODEL_NAME, metadata["model"]
            )
        if "provider" in metadata:
            current_span.set_attribute(
                AttributeKeys.JUDGMENT_LLM_PROVIDER, metadata["provider"]
            )

        if "non_cached_input_tokens" in metadata:
            current_span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                metadata["non_cached_input_tokens"],
            )
        if "output_tokens" in metadata:
            current_span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, metadata["output_tokens"]
            )
        if "cache_read_input_tokens" in metadata:
            current_span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS,
                metadata["cache_read_input_tokens"],
            )
        if "cache_creation_input_tokens" in metadata:
            current_span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
                metadata["cache_creation_input_tokens"],
            )
        if "total_cost_usd" in metadata:
            current_span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_TOTAL_COST_USD, metadata["total_cost_usd"]
            )

    # ------------------------------------------------------------------ #
    #  Static: Context Propagation                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_propagating_baggage_key(key: str, value: str) -> None:
        """
        Helper utility for the general practice of setting a key on the current span, and setting it on baggage,
        such that it gets propagated to all child spans. This will also reattach the context to the new context
        with updated baggage.
        """
        proxy = BaseTracer._get_proxy_provider()
        current_span = proxy.get_current_span()
        if current_span is None or not current_span.is_recording():
            return
        current_span.set_attribute(key, value)
        ctx = baggage.set_baggage(key, value, proxy.get_current_context())
        proxy.attach_context(ctx)

    @staticmethod
    def set_customer_id(customer_id: str) -> None:
        """Associate the current trace with a customer.

        Once set, this ID propagates to all child spans and enables
        per-customer analytics in the Judgment dashboard. Call this
        early in your request handler.

        Args:
            customer_id: Your internal customer identifier.

        Examples:
            ```python
            @Tracer.observe(span_type="agent")
            def handle_request(user_id: str, question: str):
                Tracer.set_customer_id(user_id)
                return answer(question)
            ```
        """
        BaseTracer._set_propagating_baggage_key(
            AttributeKeys.JUDGMENT_CUSTOMER_ID.value, customer_id
        )

    @staticmethod
    def set_customer_user_id(customer_user_id: str) -> None:
        """Set the customer user ID on the current span and propagate to children.

        Args:
            customer_user_id: The customer user ID to associate with this
                trace.
        """
        BaseTracer._set_propagating_baggage_key(
            AttributeKeys.JUDGMENT_CUSTOMER_USER_ID.value, customer_user_id
        )

    @staticmethod
    def set_session_id(session_id: str) -> None:
        """Associate the current trace with a conversation session.

        Groups multiple requests into a session in the Judgment dashboard.
        Propagates to all child spans. Call this early in your request handler.

        Args:
            session_id: Your session or conversation identifier.

        Examples:
            ```python
            @Tracer.observe(span_type="agent")
            def handle_message(session_id: str, message: str):
                Tracer.set_session_id(session_id)
                return chatbot.respond(message)
            ```
        """
        BaseTracer._set_propagating_baggage_key(
            AttributeKeys.JUDGMENT_SESSION_ID.value, session_id
        )

    # ------------------------------------------------------------------ #
    #  Static: Tags                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    @debug_time
    @dont_throw
    def tag(tags: str | list[str]) -> None:
        """Add tags to the current trace for filtering in the dashboard.

        Tags are sent asynchronously and appear in the Judgment monitoring
        view. Useful for marking traces by feature, experiment, or user segment.

        Args:
            tags: A single tag string or a list of tags.

        Examples:
            ```python
            Tracer.tag("rag-pipeline")
            Tracer.tag(["experiment-v2", "premium-user"])
            ```
        """
        if not tags or (isinstance(tags, list) and len(tags) == 0):
            return
        proxy = BaseTracer._get_proxy_provider()
        tracer = proxy.get_active_tracer()
        if not tracer or not tracer.project_id:
            return
        ids = BaseTracer._get_current_trace_and_span_id()
        if not ids:
            return
        client = tracer._client
        if not client:
            return
        project_id = tracer.project_id
        trace_id = ids[0]
        tag_list = tags if isinstance(tags, list) else [tags]
        bg_enqueue(
            lambda: client.post_projects_traces_by_trace_id_tags(
                project_id=project_id,
                trace_id=trace_id,
                payload={"tags": tag_list},
            )
        )

    # ------------------------------------------------------------------ #
    #  Static API: Async Evaluation                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    @debug_time
    @dont_throw
    def async_evaluate(judge: str, example: Optional[Dict[str, Any]] = None) -> None:
        """Run a hosted evaluation on this span when it completes.

        The evaluation is queued and processed server-side by the Judgment
        platform after the span ends. Use this to score live traffic
        without blocking your application.

        Args:
            judge: Name of the hosted judge/scorer (e.g. `"faithfulness"`,
                `"answer_relevancy"`).
            example: Optional dict with evaluation data. Keys like `input`,
                `actual_output`, `expected_output`, and `retrieval_context`
                are commonly used.

        Examples:
            ```python
            @Tracer.observe(span_type="agent")
            def answer(question: str) -> str:
                response = llm.generate(question)
                Tracer.async_evaluate(
                    "faithfulness",
                    {"input": question, "actual_output": response},
                )
                return response
            ```
        """
        proxy = BaseTracer._get_proxy_provider()
        tracer = proxy.get_active_tracer()
        if not tracer or not tracer.project_id:
            return
        current_span = proxy.get_current_span()
        if current_span is None or not current_span.is_recording():
            return

        processor = tracer.get_span_processor()

        ctx = current_span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")

        idx = processor.state_incr(ctx, InternalAttributeKeys.PENDING_EVALS_COUNT)
        payload = {
            "project_id": tracer.project_id,
            "eval_name": f"async_evaluate_{judge}_{idx}",
            "judges": [{"name": judge}],
            "examples": [
                {
                    **(example or {}),
                    "example_id": str(uuid4()),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "trace_id": trace_id,
                    "span_id": span_id,
                },
            ],
            "is_offline": False,
            "is_behavior": False,
        }
        updated = processor.state_append(
            ctx, InternalAttributeKeys.PENDING_EVALS, payload
        )

        current_span.set_attribute(
            AttributeKeys.JUDGMENT_PENDING_TRACE_EVAL,
            json.dumps(updated),
        )


def _format_inputs(
    f: Callable[..., Any], args: tuple, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Map positional and keyword arguments back to their parameter names
    using the function's signature. Used by ``@observe`` to record
    structured input on spans."""
    try:
        params = list(inspect.signature(f).parameters.values())
        inputs: Dict[str, Any] = {}
        arg_i = 0
        for param in params:
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if arg_i < len(args):
                    inputs[param.name] = args[arg_i]
                    arg_i += 1
                elif param.name in kwargs:
                    inputs[param.name] = kwargs[param.name]
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                inputs[param.name] = args[arg_i:]
                arg_i = len(args)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                inputs[param.name] = kwargs
        return inputs
    except Exception:
        return {}
