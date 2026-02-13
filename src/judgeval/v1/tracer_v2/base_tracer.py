from __future__ import annotations

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

from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from judgeval.logger import judgeval_logger
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.decorators.debug_time import debug_time
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.utils.serialize import serialize_attribute, safe_serialize
from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
from judgeval.v1.data import Example
from judgeval.v1.tracer_v2.proxy_tracer_provider import ProxyTracerProvider
from judgeval.v1.background_queue import enqueue as bg_enqueue

if TYPE_CHECKING:
    from judgeval.v1.internal.api.api_types import (
        JudgeExampleEvaluationRun,
        JudgeTraceEvaluationRun,
    )

C = TypeVar("C", bound=Callable[..., Any])


class BaseTracer(ABC):
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

    @staticmethod
    def _get_proxy_provider() -> ProxyTracerProvider:
        return ProxyTracerProvider.get_instance()

    @staticmethod
    def registerOTELInstrumentation(instrumentor) -> None:
        proxy = BaseTracer._get_proxy_provider()
        proxy.add_instrumentation(instrumentor)

    @staticmethod
    def get_current_span() -> Span:
        proxy = BaseTracer._get_proxy_provider()
        return proxy.get_current_span()

    @staticmethod
    def flush(timeout_ms: int = 30000) -> bool:
        proxy = BaseTracer._get_proxy_provider()
        return proxy.force_flush(timeout_ms)

    @staticmethod
    def _get_serializer() -> Callable[[Any], str]:
        tracer = BaseTracer._get_proxy_provider().get_active_tracer()
        return tracer.serializer if tracer else safe_serialize

    @staticmethod
    def _get_span_ids() -> Optional[tuple[str, str]]:
        proxy = BaseTracer._get_proxy_provider()
        current_span = proxy.get_current_span()
        if current_span is None or not current_span.is_recording():
            return None
        ctx = current_span.get_span_context()
        if not ctx.is_valid or not ctx.trace_flags.sampled:
            return None
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")

    @staticmethod
    @debug_time
    @dont_throw
    def asyncEvaluate(
        judge: str,
        examples: Sequence[Example],
        /,
    ) -> None:
        proxy = BaseTracer._get_proxy_provider()
        tracer = proxy.get_active_tracer()
        if not tracer or not tracer._client or not tracer.project_id:
            judgeval_logger.warning("asyncEvaluate: no active tracer or not configured")
            return
        ids = BaseTracer._get_span_ids()
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
        proxy = BaseTracer._get_proxy_provider()
        tracer = proxy.get_active_tracer()
        if not tracer or not tracer._client or not tracer.project_id:
            judgeval_logger.warning(
                "asyncTraceEvaluate: no active tracer or not configured"
            )
            return
        ids = BaseTracer._get_span_ids()
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

    @staticmethod
    @overload
    def observe(
        func: C,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
    ) -> C: ...

    @staticmethod
    @overload
    def observe(
        func: None = None,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
    ) -> Callable[[C], C]: ...

    @staticmethod
    def observe(
        func: Optional[C] = None,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
    ) -> C | Callable[[C], C]:
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
                            result = f(*args, **kwargs)
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

                return cast(C, sync_wrapper)

        if func is None:
            return decorator
        return decorator(func)

    @abstractmethod
    def force_flush(self, timeout_millis: int) -> bool:
        pass

    @abstractmethod
    def shutdown(self, timeout_millis: int) -> None:
        pass

    @dont_throw
    def set_attribute(self, key: str, value: Any) -> None:
        current_span = self._get_proxy_provider().get_current_span()
        if current_span is None or not current_span.is_recording():
            return
        if not key or value is None:
            return
        current_span.set_attribute(key, serialize_attribute(value, self.serializer))

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        if attributes is None:
            return
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def set_input(self, input_data: Any) -> None:
        self.set_attribute(AttributeKeys.JUDGMENT_INPUT, input_data)

    def set_output(self, output_data: Any) -> None:
        self.set_attribute(AttributeKeys.JUDGMENT_OUTPUT, output_data)

    @contextmanager
    def span(self, span_name: str) -> Iterator[Span]:
        proxy = self._get_proxy_provider()
        tracer = proxy.get_tracer(JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME)
        with tracer.start_as_current_span(span_name) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


def _format_inputs(
    f: Callable[..., Any], args: tuple, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
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
