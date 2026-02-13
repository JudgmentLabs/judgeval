from __future__ import annotations

import contextvars
import functools
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    cast,
    overload,
)
from opentelemetry.context import set_value
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider

from judgeval.logger import judgeval_logger
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.decorators.debug_time import debug_time
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.utils.serialize import serialize_attribute, safe_serialize
from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
from judgeval.v1.data import Example
from judgeval.v1.trace.proxy_tracer_provider import ProxyTracerProvider
from judgeval.v1.trace.processors._lifecycles import (
    CUSTOMER_ID_KEY,
    SESSION_ID_KEY,
)
from judgeval.v1.trace.generators import (
    _ObservedSyncGenerator,
    _ObservedAsyncGenerator,
)
from judgeval.v1.background_queue import enqueue as bg_enqueue

if TYPE_CHECKING:
    from judgeval.v1.internal.api.api_types import (
        JudgeExampleEvaluationRun,
        JudgeTraceEvaluationRun,
    )

C = TypeVar("C", bound=Callable[..., Any])


class BaseTracer(ABC):
    """Abstract base for all Judgment tracers.

    Provides the core tracing surface: span creation, attribute recording,
    the ``@observe`` and ``@agent`` decorators, context propagation for
    customer/session IDs, tagging, and async evaluation dispatch.
    Concrete subclasses supply the OTel TracerProvider, exporter, and
    processor wiring.
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
    ):
        self.project_name = project_name
        self.project_id = project_id
        self.api_key = api_key
        self.organization_id = organization_id
        self.api_url = api_url
        self.environment = environment
        self.serializer = serializer
        self._tracer_provider = tracer_provider

    # ------------------------------------------------------------------ #
    #  Abstract Lifecycle                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def force_flush(self, timeout_millis: int) -> bool:
        """Flush pending spans to the exporter within the given timeout."""

    @abstractmethod
    def shutdown(self, timeout_millis: int) -> None:
        """Shut down the tracer provider and release resources."""

    # ------------------------------------------------------------------ #
    #  Internal Helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_proxy_provider() -> ProxyTracerProvider:
        return ProxyTracerProvider.get_instance()

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
        processor = getattr(tracer, "_span_processor", None)
        if processor is not None and hasattr(processor, "emit_partial"):
            processor.emit_partial()

    # ------------------------------------------------------------------ #
    #  Static API: Span Access & Flushing                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_current_span() -> Span:
        proxy = BaseTracer._get_proxy_provider()
        return proxy.get_current_span()

    @staticmethod
    def flush(timeout_ms: int = 30000) -> bool:
        proxy = BaseTracer._get_proxy_provider()
        return proxy.force_flush(timeout_ms)

    @staticmethod
    def registerOTELInstrumentation(instrumentor) -> None:
        """Register a third-party OTel instrumentor so its spans are
        routed through the Judgment trace pipeline."""
        proxy = BaseTracer._get_proxy_provider()
        proxy.add_instrumentation(instrumentor)

    # ------------------------------------------------------------------ #
    #  Static API: Observation Decorator                                  #
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
        """Wrap a sync or async function in an OTel span.

        Supports ``@observe``, ``@observe()``, and ``@observe(span_type="tool")``
        usage. Handles sync generators and async generators by keeping the
        span open until the generator is exhausted.

        Args:
            func: The function to wrap (provided implicitly for bare decorator usage).
            span_type: Value set as the ``judgment.span_kind`` attribute.
            span_name: Span name override; defaults to ``func.__name__``.
            record_input: Whether to serialize and record function inputs.
            record_output: Whether to serialize and record the return value.
            disable_generator_yield_span: When True, suppresses per-yield child spans.
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

    # ------------------------------------------------------------------ #
    #  Static API: Async Evaluation                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    @debug_time
    @dont_throw
    def asyncEvaluate(
        judge: str,
        examples: Sequence[Example],
        /,
    ) -> None:
        """Enqueue example-level evaluation against *judge* on the background queue.

        Requires an active tracer with a configured client and project, and a
        currently recording span to bind the evaluation to.
        """
        proxy = BaseTracer._get_proxy_provider()
        tracer = proxy.get_active_tracer()
        if not tracer or not tracer._client or not tracer.project_id:
            judgeval_logger.warning("asyncEvaluate: no active tracer or not configured")
            return
        ids = BaseTracer._get_current_trace_and_span_id()
        if not ids:
            judgeval_logger.warning("asyncEvaluate: no active span")
            return
        client = tracer._client
        project_id = tracer.project_id
        payload: JudgeExampleEvaluationRun = {
            "eval_name": f"async_evaluate_{ids[1]}",
            "judge_names": [judge],
            "examples": [example.to_dict() for example in examples],
        }
        bg_enqueue(
            lambda: client.post_projects_eval_queue_judge_examples(project_id, payload)
        )

    @staticmethod
    @debug_time
    @dont_throw
    def asyncTraceEvaluate(judge: str, /) -> None:
        """Enqueue trace-level evaluation against *judge* on the background queue.

        Binds to the current trace and span IDs so the backend can locate the
        full trace for judgment.
        """
        proxy = BaseTracer._get_proxy_provider()
        tracer = proxy.get_active_tracer()
        if not tracer or not tracer._client or not tracer.project_id:
            judgeval_logger.warning(
                "asyncTraceEvaluate: no active tracer or not configured"
            )
            return
        ids = BaseTracer._get_current_trace_and_span_id()
        if not ids:
            judgeval_logger.warning("asyncTraceEvaluate: no active span")
            return
        client = tracer._client
        project_id = tracer.project_id
        payload: JudgeTraceEvaluationRun = {
            "eval_name": f"async_trace_evaluate_{ids[1]}",
            "judge_names": [judge],
            "trace_and_span_ids": [[ids[0], ids[1]]],
        }
        bg_enqueue(
            lambda: client.post_projects_eval_queue_judge_traces(project_id, payload)
        )

    # ------------------------------------------------------------------ #
    #  Static: Span Kind                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
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
        """Set a single serialized attribute on the current span."""
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
        """Set multiple attributes on the current span."""
        if attributes is None:
            return
        for key, value in attributes.items():
            BaseTracer.set_attribute(key, value)

    @staticmethod
    def set_input(input_data: Any) -> None:
        """Set the ``judgment.input`` attribute on the current span."""
        BaseTracer.set_attribute(AttributeKeys.JUDGMENT_INPUT, input_data)

    @staticmethod
    def set_output(output_data: Any) -> None:
        """Set the ``judgment.output`` attribute on the current span."""
        BaseTracer.set_attribute(AttributeKeys.JUDGMENT_OUTPUT, output_data)

    # ------------------------------------------------------------------ #
    #  Static: Context Propagation                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def set_customer_id(customer_id: str) -> None:
        """Set the customer ID on the current span and propagate it
        through the OTel context so child spans inherit it."""
        proxy = BaseTracer._get_proxy_provider()
        current_span = proxy.get_current_span()
        if current_span is None or not current_span.is_recording():
            return
        current_span.set_attribute(AttributeKeys.JUDGMENT_CUSTOMER_ID, customer_id)
        ctx = set_value(CUSTOMER_ID_KEY, customer_id, proxy.get_current_context())
        proxy.attach_context(ctx)

    @staticmethod
    def set_session_id(session_id: str) -> None:
        """Set the session ID on the current span and propagate it
        through the OTel context so child spans inherit it.
        Only applies to valid, sampled spans."""
        proxy = BaseTracer._get_proxy_provider()
        current_span = proxy.get_current_span()
        if current_span is None or not current_span.is_recording():
            return
        span_ctx = current_span.get_span_context()
        if not span_ctx.is_valid or not span_ctx.trace_flags.sampled:
            return
        current_span.set_attribute(AttributeKeys.JUDGMENT_SESSION_ID, session_id)
        ctx = set_value(SESSION_ID_KEY, session_id, proxy.get_current_context())
        proxy.attach_context(ctx)

    # ------------------------------------------------------------------ #
    #  Static: Tags                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    @debug_time
    @dont_throw
    def tag(tags: str | list[str]) -> None:
        """Attach one or more tags to the current trace via the API."""
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
    #  Static: Span Context Manager                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    @contextmanager
    def span(span_name: str) -> Iterator[Span]:
        """Open a child span under the current trace context.
        Exceptions propagate after being recorded on the span."""
        proxy = BaseTracer._get_proxy_provider()
        tracer = proxy.get_tracer(JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME)
        with tracer.start_as_current_span(span_name) as span:
            BaseTracer._emit_partial()
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


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
