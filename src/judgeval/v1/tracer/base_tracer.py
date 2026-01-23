from __future__ import annotations

import datetime
import functools
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    cast,
    overload,
)

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span, SpanContext, Status, StatusCode
from opentelemetry import context as otel_context
from opentelemetry.context.context import Context

from judgeval.logger import judgeval_logger
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.v1.data.example import Example
from judgeval.v1.instrumentation import wrap_provider
from judgeval.v1.instrumentation.llm.providers import ApiClient
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.utils.url import url_for
from judgeval.v1.utils import resolve_project_id
from judgeval.v1.internal.api.api_types import (
    ExampleEvaluationRun,
    TraceEvaluationRun,
)
from judgeval.v1.scorers.base_scorer import BaseScorer
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer
from judgeval.v1.tracer.exporters.judgment_span_exporter import JudgmentSpanExporter
from judgeval.v1.tracer.processors.judgment_span_processor import JudgmentSpanProcessor
from uuid import uuid4
from opentelemetry.context import attach, detach, get_value, set_value
from judgeval.v1.tracer.processors._lifecycles import (
    AGENT_ID_KEY,
    PARENT_AGENT_ID_KEY,
    CUSTOMER_ID_KEY,
    SESSION_ID_KEY,
    AGENT_CLASS_NAME_KEY,
    AGENT_INSTANCE_NAME_KEY,
)
from judgeval.v1.tracer.isolated.tracer import JudgmentIsolatedTracer

C = TypeVar("C", bound=Callable[..., Any])
T = TypeVar("T", bound=ApiClient)


class BaseTracer(ABC):
    __slots__ = (
        "project_name",
        "enable_evaluation",
        "enable_monitoring",
        "api_client",
        "serializer",
        "project_id",
        "_tracer_provider",
    )

    TRACER_NAME = "judgeval"
    _tracers: list[BaseTracer] = []

    def __init__(
        self,
        project_name: str,
        enable_evaluation: bool,
        enable_monitoring: bool,
        api_client: JudgmentSyncClient,
        serializer: Callable[[Any], str],
        tracer_provider: TracerProvider,
    ):
        self.project_name = project_name
        self.enable_evaluation = enable_evaluation
        self.enable_monitoring = enable_monitoring
        self.api_client = api_client
        self.serializer = serializer
        self.project_id = resolve_project_id(api_client, project_name)
        self._tracer_provider = tracer_provider

        BaseTracer._tracers.append(self)

        if self.project_id is None:
            judgeval_logger.error(
                f"Failed to resolve project {project_name}, "
                f"please create it first at https://app.judgmentlabs.ai/org/{self.api_client.organization_id}/projects. "
                "Skipping Judgment export."
            )

    @abstractmethod
    def force_flush(self, timeout_millis: int) -> bool:
        pass

    @abstractmethod
    def shutdown(self, timeout_millis: int) -> None:
        pass

    def get_span_exporter(self) -> SpanExporter:
        if self.project_id is not None:
            return JudgmentSpanExporter(
                endpoint=self._build_endpoint(self.api_client.base_url),
                api_key=self.api_client.api_key,
                organization_id=self.api_client.organization_id,
                project_id=self.project_id,
            )
        else:
            judgeval_logger.error(
                "Project not resolved; cannot create exporter, returning NoOpSpanExporter"
            )
            from judgeval.v1.tracer.exporters.noop_span_exporter import NoOpSpanExporter

            return NoOpSpanExporter()

    def get_span_processor(self) -> JudgmentSpanProcessor:
        if self.project_id is not None:
            return JudgmentSpanProcessor(
                self,
                self.get_span_exporter(),
            )
        else:
            judgeval_logger.error(
                "Project not resolved; cannot create processor, returning NoOpSpanProcessor"
            )
            from judgeval.v1.tracer.processors.noop_span_processor import (
                NoOpJudgmentSpanProcessor,
            )

            return NoOpJudgmentSpanProcessor()

    def get_tracer(self) -> trace.Tracer:
        return self._tracer_provider.get_tracer(self.TRACER_NAME)

    @property
    def tracer_provider(self) -> TracerProvider:
        return self._tracer_provider

    def _is_isolated(self) -> bool:
        from judgeval.v1.tracer.judgment_tracer_provider import JudgmentTracerProvider

        if isinstance(self._tracer_provider, JudgmentTracerProvider):
            return self._tracer_provider._isolated
        return False

    def get_context(self) -> Context:
        from judgeval.v1.tracer.judgment_tracer_provider import JudgmentTracerProvider

        if self._is_isolated() and isinstance(
            self._tracer_provider, JudgmentTracerProvider
        ):
            return self._tracer_provider.get_isolated_current_context()
        return otel_context.get_current()

    def use_span(
        self,
        span: Span,
        end_on_exit: bool = False,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ):
        if self._is_isolated():
            tracer = cast(JudgmentIsolatedTracer, self.get_tracer())
            return tracer.use_span(
                span,
                end_on_exit=end_on_exit,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            )
        else:
            return trace.use_span(
                span,
                end_on_exit=end_on_exit,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            )

    def _get_current_span(self) -> Optional[Span]:
        if self._is_isolated():
            tracer = cast(JudgmentIsolatedTracer, self.get_tracer())
            return tracer.get_current_span()
        return trace.get_current_span()

    def set_span_kind(self, kind: str) -> None:
        if kind is None:
            return
        current_span = self._get_current_span()
        if current_span is not None:
            current_span.set_attribute(AttributeKeys.JUDGMENT_SPAN_KIND, kind)

    @dont_throw
    def set_attribute(self, key: str, value: Any) -> None:
        if not self._is_valid_key(key):
            return
        if value is None:
            return
        current_span = self._get_current_span()
        if current_span is not None:
            serialized_value = (
                self.serializer(value)
                if not isinstance(value, (str, int, float, bool))
                else value
            )
            current_span.set_attribute(key, serialized_value)

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        if attributes is None:
            return
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def set_customer_id(self, customer_id: str) -> None:
        current_span = self._get_current_span()
        if current_span is None:
            return
        current_span.set_attribute(AttributeKeys.JUDGMENT_CUSTOMER_ID, customer_id)
        ctx = set_value(CUSTOMER_ID_KEY, customer_id)
        attach(ctx)

    def set_llm_span(self) -> None:
        self.set_span_kind("llm")

    def set_tool_span(self) -> None:
        self.set_span_kind("tool")

    def set_general_span(self) -> None:
        self.set_span_kind("span")

    def set_input(self, input_data: Any) -> None:
        self.set_attribute(AttributeKeys.JUDGMENT_INPUT, input_data)

    def set_output(self, output_data: Any) -> None:
        self.set_attribute(AttributeKeys.JUDGMENT_OUTPUT, output_data)

    @contextmanager
    def span(self, span_name: str) -> Iterator[Span]:
        tracer = self.get_tracer()
        with tracer.start_as_current_span(span_name) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def start_span(self, span_name: str) -> Span:
        tracer = self.get_tracer()
        return tracer.start_span(span_name)

    @dont_throw
    def async_evaluate(
        self,
        scorer: BaseScorer | None,
        example: Example,
    ) -> None:
        if scorer is None:
            judgeval_logger.error("Scorer was not found, skipping evaluation.")
            return

        if not self.enable_evaluation:
            return

        span_context = self._get_sampled_span_context()
        if span_context is None:
            return

        trace_id = span_context.trace_id
        span_id = span_context.span_id
        trace_id_hex = format(trace_id, "032x")
        span_id_hex = format(span_id, "016x")

        self._log_evaluation_info(
            "asyncEvaluate", trace_id_hex, span_id_hex, scorer.get_name()
        )

        evaluation_run = self._create_evaluation_run(
            scorer, example, trace_id_hex, span_id_hex
        )
        self._enqueue_evaluation(evaluation_run)

    @dont_throw
    def async_trace_evaluate(
        self,
        scorer: BaseScorer | None,
    ) -> None:
        if scorer is None:
            judgeval_logger.error("Scorer was not found, skipping trace evaluation.")
            return

        if not self.enable_evaluation:
            return

        current_span = self._get_sampled_span()
        if current_span is None:
            return

        span_context = current_span.get_span_context()
        trace_id = span_context.trace_id
        span_id = span_context.span_id
        trace_id_hex = format(trace_id, "032x")
        span_id_hex = format(span_id, "016x")

        self._log_evaluation_info(
            "asyncTraceEvaluate", trace_id_hex, span_id_hex, scorer.get_name()
        )

        evaluation_run = self._create_trace_evaluation_run(
            scorer, trace_id_hex, span_id_hex
        )
        try:
            trace_eval_json = self.serializer(evaluation_run)
            current_span.set_attribute(
                AttributeKeys.JUDGMENT_PENDING_TRACE_EVAL, trace_eval_json
            )
        except Exception as e:
            judgeval_logger.error(f"Failed to serialize trace evaluation: {e}")

    def set_session_id(self, session_id: str) -> None:
        current_span = self._get_sampled_span()
        if current_span is None:
            return
        current_span.set_attribute(AttributeKeys.JUDGMENT_SESSION_ID, session_id)
        ctx = set_value(SESSION_ID_KEY, session_id)
        attach(ctx)

    @dont_throw
    def tag(self, tags: str | list[str]) -> None:
        # NOTE: Manual API call until PR #661 is merged, then will use generated client
        if not tags or (isinstance(tags, list) and len(tags) == 0):
            return
        span_context = self._get_sampled_span_context()
        if span_context is None:
            return
        trace_id = format(span_context.trace_id, "032x")
        self.api_client._request(
            "POST",
            url_for("/traces/tags/add", self.api_client.base_url),
            {
                "project_name": self.project_name,
                "trace_id": trace_id,
                "tags": tags if isinstance(tags, list) else [tags],
            },
        )

    def _build_endpoint(self, base_url: str) -> str:
        return (
            base_url + "otel/v1/traces"
            if base_url.endswith("/")
            else base_url + "/otel/v1/traces"
        )

    def _generate_run_id(self, prefix: str, span_id: str) -> str:
        return prefix + span_id

    def _create_evaluation_run(
        self,
        scorer: BaseScorer,
        example: Example,
        trace_id: str,
        span_id: str,
    ) -> ExampleEvaluationRun:
        run_id = self._generate_run_id("async_evaluate_", span_id)

        judgment_scorers = (
            [] if isinstance(scorer, CustomScorer) else [scorer.get_scorer_config()]
        )
        custom_scorers = [scorer.to_dict()] if isinstance(scorer, CustomScorer) else []

        return ExampleEvaluationRun(
            project_name=self.project_name,
            eval_name=run_id,
            trace_id=trace_id,
            trace_span_id=span_id,
            examples=[example.to_dict()],
            judgment_scorers=judgment_scorers,
            custom_scorers=custom_scorers,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def _create_trace_evaluation_run(
        self,
        scorer: BaseScorer,
        trace_id: str,
        span_id: str,
    ) -> TraceEvaluationRun:
        eval_name = self._generate_run_id("async_trace_evaluate_", span_id)

        judgment_scorers = (
            [] if isinstance(scorer, CustomScorer) else [scorer.get_scorer_config()]
        )
        custom_scorers = [scorer.to_dict()] if isinstance(scorer, CustomScorer) else []

        return TraceEvaluationRun(
            project_name=self.project_name,
            eval_name=eval_name,
            trace_and_span_ids=[[trace_id, span_id]],
            judgment_scorers=judgment_scorers,
            custom_scorers=custom_scorers,
            is_offline=False,
            is_bucket_run=False,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def _enqueue_evaluation(self, evaluation_run: ExampleEvaluationRun) -> None:
        try:
            self.api_client.add_to_run_eval_queue_examples(evaluation_run)
        except Exception as e:
            judgeval_logger.error(f"Failed to enqueue evaluation run: {e}")

    def _get_sampled_span_context(self) -> Optional[SpanContext]:
        current_span = self._get_current_span()
        if current_span is None:
            return None
        span_context = current_span.get_span_context()
        if not span_context.is_valid or not span_context.trace_flags.sampled:
            return None
        return span_context

    def _get_sampled_span(self) -> Optional[Span]:
        current_span = self._get_current_span()
        if current_span is None:
            return None
        span_context = current_span.get_span_context()
        if not span_context.is_valid or not span_context.trace_flags.sampled:
            return None
        return current_span

    def _log_evaluation_info(
        self, method: str, trace_id: str, span_id: str, scorer_name: str
    ) -> None:
        judgeval_logger.info(
            f"{method}: project={self.project_name}, traceId={trace_id}, spanId={span_id}, scorer={scorer_name}"
        )

    @staticmethod
    def _is_valid_key(key: str) -> bool:
        return key is not None and len(key) > 0

    @overload
    def observe(
        self,
        func: C,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
    ) -> C: ...

    @overload
    def observe(
        self,
        func: None = None,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
    ) -> Callable[[C], C]: ...

    def observe(
        self,
        func: Optional[C] = None,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        record_input: bool = True,
        record_output: bool = True,
    ) -> C | Callable[[C], C]:
        if func is None:
            return lambda f: self.observe(
                f, span_type, span_name, record_input, record_output
            )  # type: ignore[return-value]

        if not self.enable_monitoring:
            judgeval_logger.info(
                "Monitoring disabled, observe() returning function unchanged"
            )
            return func

        tracer = self.get_tracer()
        name = span_name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(name) as span:
                    if span_type:
                        span.set_attribute(AttributeKeys.JUDGMENT_SPAN_KIND, span_type)

                    try:
                        if record_input:
                            input_data = _format_inputs(func, args, kwargs)
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_INPUT,
                                self.serializer(input_data),
                            )

                        result = await func(*args, **kwargs)

                        if record_output:
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_OUTPUT, self.serializer(result)
                            )
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(name) as span:
                    if span_type:
                        span.set_attribute(AttributeKeys.JUDGMENT_SPAN_KIND, span_type)

                    try:
                        if record_input:
                            input_data = _format_inputs(func, args, kwargs)
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_INPUT,
                                self.serializer(input_data),
                            )

                        result = func(*args, **kwargs)

                        if record_output:
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_OUTPUT, self.serializer(result)
                            )
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return sync_wrapper  # type: ignore[return-value]

    @overload
    def agent(self, func: C, /, *, identifier: Optional[str] = None) -> C: ...

    @overload
    def agent(
        self, func: None = None, /, *, identifier: Optional[str] = None
    ) -> Callable[[C], C]: ...

    def agent(
        self, func: Optional[C] = None, /, *, identifier: Optional[str] = None
    ) -> C | Callable[[C], C]:
        if func is None:
            return lambda f: self.agent(f, identifier=identifier)  # type: ignore[return-value]

        class_name = None
        if hasattr(func, "__qualname__") and "." in func.__qualname__:
            parts = func.__qualname__.split(".")
            if len(parts) >= 2:
                class_name = parts[-2]

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                agent_id = str(uuid4())
                parent_agent_id = get_value(AGENT_ID_KEY)
                ctx = set_value(AGENT_ID_KEY, agent_id)
                if parent_agent_id:
                    ctx = set_value(PARENT_AGENT_ID_KEY, parent_agent_id, context=ctx)
                if class_name:
                    ctx = set_value(AGENT_CLASS_NAME_KEY, class_name, context=ctx)
                if identifier and args:
                    instance = args[0]
                    if hasattr(instance, identifier):
                        instance_name = str(getattr(instance, identifier))
                        ctx = set_value(
                            AGENT_INSTANCE_NAME_KEY, instance_name, context=ctx
                        )
                token = attach(ctx)
                try:
                    return await func(*args, **kwargs)
                finally:
                    detach(token)

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                agent_id = str(uuid4())
                parent_agent_id = get_value(AGENT_ID_KEY)
                ctx = set_value(AGENT_ID_KEY, agent_id)
                if parent_agent_id:
                    ctx = set_value(PARENT_AGENT_ID_KEY, parent_agent_id, context=ctx)
                if class_name:
                    ctx = set_value(AGENT_CLASS_NAME_KEY, class_name, context=ctx)
                if identifier and args:
                    instance = args[0]
                    if hasattr(instance, identifier):
                        instance_name = str(getattr(instance, identifier))
                        ctx = set_value(
                            AGENT_INSTANCE_NAME_KEY, instance_name, context=ctx
                        )
                token = attach(ctx)
                try:
                    return func(*args, **kwargs)
                finally:
                    detach(token)

            return sync_wrapper  # type: ignore[return-value]

    def wrap(self, client: T) -> T:
        return wrap_provider(self, client)


def _format_inputs(
    f: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
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
