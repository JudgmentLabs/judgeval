from __future__ import annotations
import os
from contextvars import ContextVar
import atexit
import functools
import inspect
import random
from typing import (
    Any,
    Union,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    overload,
    Literal,
    TypedDict,
    Iterator,
    AsyncIterator,
)
from functools import partial
from warnings import warn

from opentelemetry.sdk.trace import SpanProcessor, TracerProvider, Span
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import (
    Status,
    StatusCode,
    TracerProvider as ABCTracerProvider,
    NoOpTracerProvider,
    Tracer as ABCTracer,
    get_current_span,
)

from judgeval.data.evaluation_run import ExampleEvaluationRun, TraceEvaluationRun
from judgeval.data.example import Example
from judgeval.env import (
    JUDGMENT_API_KEY,
    JUDGMENT_DEFAULT_GPT_MODEL,
    JUDGMENT_ORG_ID,
)
from judgeval.logger import judgeval_logger
from judgeval.scorers.api_scorer import ExampleAPIScorerConfig, TraceAPIScorerConfig
from judgeval.scorers.base_scorer import BaseScorer
from judgeval.tracer.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
from judgeval.tracer.managers import (
    sync_span_context,
    async_span_context,
    sync_agent_context,
    async_agent_context,
)
from judgeval.utils.serialize import safe_serialize
from judgeval.version import get_version
from judgeval.warnings import JudgmentWarning

from judgeval.tracer.keys import AttributeKeys, InternalAttributeKeys
from judgeval.api import JudgmentSyncClient
from judgeval.tracer.llm import wrap_provider
from judgeval.utils.url import url_for
from judgeval.tracer.local_eval_queue import LocalEvaluationQueue
from judgeval.tracer.processors import (
    JudgmentSpanProcessor,
    NoOpJudgmentSpanProcessor,
    NoOpSpanProcessor,
)
from judgeval.tracer.utils import set_span_attribute, TraceScorerConfig

C = TypeVar("C", bound=Callable)
Cls = TypeVar("Cls", bound=Type)
ApiClient = TypeVar("ApiClient", bound=Any)


class AgentContext(TypedDict):
    agent_id: str
    class_name: str | None
    instance_name: str | None
    track_state: bool
    track_attributes: List[str] | None
    field_mappings: Dict[str, str]
    instance: Any
    is_agent_entry_point: bool
    parent_agent_id: str | None


class Tracer:
    _active_tracers: List[Tracer] = []

    __slots__ = (
        "api_key",
        "organization_id",
        "project_name",
        "api_url",
        "deep_tracing",
        "enable_monitoring",
        "enable_evaluation",
        "api_client",
        "local_eval_queue",
        # Otel
        "judgment_processor",
        "processors",
        "provider",
        "tracer",
        # Agent
        "agent_context",
        "cost_context",
    )

    api_key: str
    organization_id: str
    project_name: str
    api_url: str
    deep_tracing: bool
    enable_monitoring: bool
    enable_evaluation: bool
    api_client: JudgmentSyncClient
    local_eval_queue: LocalEvaluationQueue

    judgment_processor: JudgmentSpanProcessor
    processors: List[SpanProcessor]
    provider: ABCTracerProvider
    tracer: ABCTracer

    agent_context: ContextVar[Optional[AgentContext]]
    cost_context: ContextVar[Optional[Dict[str, float]]]

    def __init__(
        self,
        /,
        *,
        project_name: str,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        deep_tracing: bool = False,
        enable_monitoring: bool = os.getenv(
            "JUDGMENT_ENABLE_MONITORING", "true"
        ).lower()
        != "false",
        enable_evaluation: bool = os.getenv(
            "JUDGMENT_ENABLE_EVALUATIONS", "true"
        ).lower()
        != "false",
        processors: List[SpanProcessor] = [],
        resource_attributes: Optional[Dict[str, Any]] = None,
    ):
        _api_key = api_key or JUDGMENT_API_KEY
        _organization_id = organization_id or JUDGMENT_ORG_ID

        if _api_key is None:
            raise ValueError(
                "API Key is not set, please set it in the environment variables or pass it as `api_key`"
            )

        if _organization_id is None:
            raise ValueError(
                "Organization ID is not set, please set it in the environment variables or pass it as `organization_id`"
            )

        self.api_key = _api_key
        self.organization_id = _organization_id
        self.project_name = project_name
        self.api_url = url_for("/otel/v1/traces")

        self.deep_tracing = deep_tracing
        self.enable_monitoring = enable_monitoring
        self.enable_evaluation = enable_evaluation

        self.judgment_processor = NoOpJudgmentSpanProcessor()
        self.processors = processors
        self.provider = NoOpTracerProvider()

        self.agent_context = ContextVar("current_agent_context", default=None)
        self.cost_context = ContextVar("current_cost_context", default=None)

        if self.enable_monitoring:
            self.judgment_processor = JudgmentSpanProcessor(
                self,
                self.project_name,
                self.api_key,
                self.organization_id,
                max_queue_size=2**18,
                export_timeout_millis=30000,
                resource_attributes=resource_attributes,
            )

            resource = Resource.create(self.judgment_processor.resource_attributes)
            self.provider = TracerProvider(resource=resource)

            self.processors.append(self.judgment_processor)
            for processor in self.processors:
                self.provider.add_span_processor(processor)

        self.tracer = self.provider.get_tracer(
            JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME,
            get_version(),
        )
        self.api_client = JudgmentSyncClient(
            api_key=self.api_key,
            organization_id=self.organization_id,
        )
        self.local_eval_queue = LocalEvaluationQueue()

        if self.enable_evaluation and self.enable_monitoring:
            self.local_eval_queue.start_workers()

        Tracer._active_tracers.append(self)

        # Register atexit handler to flush on program exit
        atexit.register(self._atexit_flush)

    def get_current_span(self):
        return get_current_span()

    def get_tracer(self):
        return self.tracer

    def get_current_agent_context(self):
        return self.agent_context

    def get_current_cost_context(self):
        return self.cost_context

    def get_processor(self):
        """Get the judgment span processor instance.

        Returns:
            The JudgmentSpanProcessor or NoOpJudgmentSpanProcessor instance used by this tracer.
        """
        return self.judgment_processor

    def set_customer_id(self, customer_id: str) -> None:
        span = self.get_current_span()
        if span and span.is_recording():
            set_span_attribute(span, AttributeKeys.JUDGMENT_CUSTOMER_ID, customer_id)

    def add_cost_to_current_context(self, cost: Optional[float]) -> None:
        """Add cost to the current cost context and update span attribute."""
        if cost is None:
            return
        current_cost_context = self.cost_context.get()
        if current_cost_context is not None:
            current_cumulative_cost = current_cost_context.get("cumulative_cost", 0.0)
            new_cumulative_cost = float(current_cumulative_cost) + cost
            current_cost_context["cumulative_cost"] = new_cumulative_cost

            span = self.get_current_span()
            if span and span.is_recording():
                set_span_attribute(
                    span,
                    AttributeKeys.JUDGMENT_CUMULATIVE_LLM_COST,
                    new_cumulative_cost,
                )

    def add_agent_attributes_to_span(self, span):
        """Add agent ID, class name, and instance name to span if they exist in context"""
        current_agent_context = self.agent_context.get()
        if not current_agent_context:
            return

        set_span_attribute(
            span, AttributeKeys.JUDGMENT_AGENT_ID, current_agent_context["agent_id"]
        )
        set_span_attribute(
            span,
            AttributeKeys.JUDGMENT_AGENT_CLASS_NAME,
            current_agent_context["class_name"],
        )
        set_span_attribute(
            span,
            AttributeKeys.JUDGMENT_AGENT_INSTANCE_NAME,
            current_agent_context["instance_name"],
        )
        set_span_attribute(
            span,
            AttributeKeys.JUDGMENT_PARENT_AGENT_ID,
            current_agent_context["parent_agent_id"],
        )
        set_span_attribute(
            span,
            AttributeKeys.JUDGMENT_IS_AGENT_ENTRY_POINT,
            current_agent_context["is_agent_entry_point"],
        )
        current_agent_context["is_agent_entry_point"] = False

    def record_instance_state(self, record_point: Literal["before", "after"], span):
        current_agent_context = self.agent_context.get()

        if current_agent_context and current_agent_context.get("track_state"):
            instance = current_agent_context.get("instance")
            track_attributes = current_agent_context.get("track_attributes")
            field_mappings = current_agent_context.get("field_mappings", {})

            if track_attributes is not None:
                attributes = {
                    field_mappings.get(attr, attr): getattr(instance, attr, None)
                    for attr in track_attributes
                }
            else:
                attributes = {
                    field_mappings.get(k, k): v
                    for k, v in instance.__dict__.items()
                    if not k.startswith("_")
                }
            set_span_attribute(
                span,
                (
                    AttributeKeys.JUDGMENT_STATE_BEFORE
                    if record_point == "before"
                    else AttributeKeys.JUDGMENT_STATE_AFTER
                ),
                safe_serialize(attributes),
            )

    def _set_pending_trace_eval(
        self,
        span: Span,
        scorer_config: TraceScorerConfig,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        if not self.enable_evaluation:
            return

        scorer = scorer_config.scorer
        model = scorer_config.model
        run_condition = scorer_config.run_condition
        sampling_rate = scorer_config.sampling_rate

        if not isinstance(scorer, (TraceAPIScorerConfig)):
            judgeval_logger.error(
                "Scorer must be an instance of TraceAPIScorerConfig, got %s, skipping evaluation."
                % type(scorer)
            )
            return

        if run_condition is not None and not run_condition(*args, **kwargs):
            return

        if sampling_rate < 0 or sampling_rate > 1:
            judgeval_logger.error(
                "Sampling rate must be between 0 and 1, got %s, skipping evaluation."
                % sampling_rate
            )
            return

        percentage = random.uniform(0, 1)
        if percentage > sampling_rate:
            judgeval_logger.info(
                "Sampling rate is %s, skipping evaluation." % sampling_rate
            )
            return

        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")
        eval_run_name = f"async_trace_evaluate_{span_id}"

        eval_run = TraceEvaluationRun(
            organization_id=self.organization_id,
            project_name=self.project_name,
            eval_name=eval_run_name,
            scorers=[scorer],
            model=model,
            trace_and_span_ids=[(trace_id, span_id)],
        )
        span.set_attribute(
            AttributeKeys.PENDING_TRACE_EVAL,
            safe_serialize(eval_run.model_dump(warnings=False)),
        )

    def _create_traced_sync_generator(
        self,
        generator: Iterator[Any],
        main_span: Span,
        base_name: str,
        attributes: Optional[Dict[str, Any]],
    ):
        """Create a traced synchronous generator that wraps each yield in a span."""
        try:
            while True:
                yield_span_name = f"{base_name}_yield"
                yield_attributes = {
                    AttributeKeys.JUDGMENT_SPAN_KIND: "generator_yield",
                    **(attributes or {}),
                }

                with sync_span_context(
                    self, yield_span_name, yield_attributes, disable_partial_emit=True
                ) as yield_span:
                    self.add_agent_attributes_to_span(yield_span)

                    try:
                        value = next(generator)
                    except StopIteration:
                        # Mark span as cancelled so it won't be exported
                        self.judgment_processor.set_internal_attribute(
                            span_context=yield_span.get_span_context(),
                            key=InternalAttributeKeys.CANCELLED,
                            value=True,
                        )
                        break

                    set_span_attribute(
                        yield_span,
                        AttributeKeys.JUDGMENT_OUTPUT,
                        safe_serialize(value),
                    )

                yield value
        except Exception as e:
            main_span.record_exception(e)
            main_span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    async def _create_traced_async_generator(
        self,
        async_generator: AsyncIterator[Any],
        main_span: Span,
        base_name: str,
        attributes: Optional[Dict[str, Any]],
    ):
        """Create a traced asynchronous generator that wraps each yield in a span."""
        try:
            while True:
                yield_span_name = f"{base_name}_yield"
                yield_attributes = {
                    AttributeKeys.JUDGMENT_SPAN_KIND: "async_generator_yield",
                    **(attributes or {}),
                }

                async with async_span_context(
                    self, yield_span_name, yield_attributes, disable_partial_emit=True
                ) as yield_span:
                    self.add_agent_attributes_to_span(yield_span)

                    try:
                        value = await async_generator.__anext__()
                    except StopAsyncIteration:
                        # Mark span as cancelled so it won't be exported
                        self.judgment_processor.set_internal_attribute(
                            span_context=yield_span.get_span_context(),
                            key=InternalAttributeKeys.CANCELLED,
                            value=True,
                        )
                        break

                    set_span_attribute(
                        yield_span,
                        AttributeKeys.JUDGMENT_OUTPUT,
                        safe_serialize(value),
                    )

                yield value
        except Exception as e:
            main_span.record_exception(e)
            main_span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    def _wrap_sync(
        self,
        f: Callable,
        name: Optional[str],
        attributes: Optional[Dict[str, Any]],
        scorer_config: TraceScorerConfig | None = None,
    ):
        # Check if this is a generator function - if so, wrap it specially
        if inspect.isgeneratorfunction(f):
            return self._wrap_sync_generator_function(
                f, name, attributes, scorer_config
            )

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            n = name or f.__qualname__
            with sync_span_context(self, n, attributes) as span:
                self.add_agent_attributes_to_span(span)
                self.record_instance_state("before", span)
                try:
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_INPUT,
                        safe_serialize(format_inputs(f, args, kwargs)),
                    )

                    self.judgment_processor.emit_partial()

                    if scorer_config:
                        self._set_pending_trace_eval(span, scorer_config, args, kwargs)

                    result = f(*args, **kwargs)
                except Exception as user_exc:
                    span.record_exception(user_exc)
                    span.set_status(Status(StatusCode.ERROR, str(user_exc)))
                    raise

                if inspect.isgenerator(result):
                    set_span_attribute(
                        span, AttributeKeys.JUDGMENT_OUTPUT, "<generator>"
                    )
                    self.record_instance_state("after", span)
                    return self._create_traced_sync_generator(
                        result, span, n, attributes
                    )
                else:
                    set_span_attribute(
                        span, AttributeKeys.JUDGMENT_OUTPUT, safe_serialize(result)
                    )
                    self.record_instance_state("after", span)
                    return result

        return wrapper

    def _wrap_sync_generator_function(
        self,
        f: Callable,
        name: Optional[str],
        attributes: Optional[Dict[str, Any]],
        scorer_config: TraceScorerConfig | None = None,
    ):
        """Wrap a generator function to trace nested function calls within each yield."""

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            n = name or f.__qualname__

            with sync_span_context(self, n, attributes) as main_span:
                self.add_agent_attributes_to_span(main_span)
                self.record_instance_state("before", main_span)

                try:
                    set_span_attribute(
                        main_span,
                        AttributeKeys.JUDGMENT_INPUT,
                        safe_serialize(format_inputs(f, args, kwargs)),
                    )

                    self.judgment_processor.emit_partial()

                    if scorer_config:
                        self._set_pending_trace_eval(
                            main_span, scorer_config, args, kwargs
                        )

                    generator = f(*args, **kwargs)
                    set_span_attribute(
                        main_span, AttributeKeys.JUDGMENT_OUTPUT, "<generator>"
                    )
                    self.record_instance_state("after", main_span)

                    return self._create_traced_sync_generator(
                        generator, main_span, n, attributes
                    )

                except Exception as user_exc:
                    main_span.record_exception(user_exc)
                    main_span.set_status(Status(StatusCode.ERROR, str(user_exc)))
                    raise

        return wrapper

    def _wrap_async(
        self,
        f: Callable,
        name: Optional[str],
        attributes: Optional[Dict[str, Any]],
        scorer_config: TraceScorerConfig | None = None,
    ):
        # Check if this is an async generator function - if so, wrap it specially
        if inspect.isasyncgenfunction(f):
            return self._wrap_async_generator_function(
                f, name, attributes, scorer_config
            )

        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            n = name or f.__qualname__
            async with async_span_context(self, n, attributes) as span:
                self.add_agent_attributes_to_span(span)
                self.record_instance_state("before", span)
                try:
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_INPUT,
                        safe_serialize(format_inputs(f, args, kwargs)),
                    )

                    self.judgment_processor.emit_partial()

                    if scorer_config:
                        self._set_pending_trace_eval(span, scorer_config, args, kwargs)

                    result = await f(*args, **kwargs)
                except Exception as user_exc:
                    span.record_exception(user_exc)
                    span.set_status(Status(StatusCode.ERROR, str(user_exc)))
                    raise

                if inspect.isasyncgen(result):
                    set_span_attribute(
                        span, AttributeKeys.JUDGMENT_OUTPUT, "<async_generator>"
                    )
                    self.record_instance_state("after", span)
                    return self._create_traced_async_generator(
                        result, span, n, attributes
                    )
                else:
                    set_span_attribute(
                        span, AttributeKeys.JUDGMENT_OUTPUT, safe_serialize(result)
                    )
                    self.record_instance_state("after", span)
                    return result

        return wrapper

    def _wrap_async_generator_function(
        self,
        f: Callable,
        name: Optional[str],
        attributes: Optional[Dict[str, Any]],
        scorer_config: TraceScorerConfig | None = None,
    ):
        """Wrap an async generator function to trace nested function calls within each yield."""

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            n = name or f.__qualname__

            with sync_span_context(self, n, attributes) as main_span:
                self.add_agent_attributes_to_span(main_span)
                self.record_instance_state("before", main_span)

                try:
                    set_span_attribute(
                        main_span,
                        AttributeKeys.JUDGMENT_INPUT,
                        safe_serialize(format_inputs(f, args, kwargs)),
                    )

                    self.judgment_processor.emit_partial()

                    if scorer_config:
                        self._set_pending_trace_eval(
                            main_span, scorer_config, args, kwargs
                        )

                    async_generator = f(*args, **kwargs)
                    set_span_attribute(
                        main_span, AttributeKeys.JUDGMENT_OUTPUT, "<async_generator>"
                    )
                    self.record_instance_state("after", main_span)

                    return self._create_traced_async_generator(
                        async_generator, main_span, n, attributes
                    )

                except Exception as user_exc:
                    main_span.record_exception(user_exc)
                    main_span.set_status(Status(StatusCode.ERROR, str(user_exc)))
                    raise

        return wrapper

    @overload
    def observe(
        self,
        func: C,
        /,
        *,
        span_type: str | None = None,
        scorer_config: TraceScorerConfig | None = None,
    ) -> C: ...

    @overload
    def observe(
        self,
        func: None = None,
        /,
        *,
        span_type: str | None = None,
        scorer_config: TraceScorerConfig | None = None,
    ) -> Callable[[C], C]: ...

    def observe(
        self,
        func: Callable | None = None,
        /,
        *,
        span_type: str | None = "span",
        span_name: str | None = None,
        attributes: Optional[Dict[str, Any]] = None,
        scorer_config: TraceScorerConfig | None = None,
    ) -> Callable | None:
        if func is None:
            return partial(
                self.observe,
                span_type=span_type,
                span_name=span_name,
                attributes=attributes,
                scorer_config=scorer_config,
            )

        if not self.enable_monitoring:
            return func

        # Handle functions (including generator functions) - detect generators at runtime
        name = span_name or getattr(func, "__qualname__", "function")
        func_attributes: Dict[str, Any] = {
            AttributeKeys.JUDGMENT_SPAN_KIND: span_type,
            **(attributes or {}),
        }

        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            return self._wrap_async(func, name, func_attributes, scorer_config)
        else:
            return self._wrap_sync(func, name, func_attributes, scorer_config)

    @overload
    def agent(
        self,
        func: C,
        /,
        *,
        identifier: str | None = None,
        track_state: bool = False,
        track_attributes: List[str] | None = None,
        field_mappings: Dict[str, str] = {},
    ) -> C: ...

    @overload
    def agent(
        self,
        func: None = None,
        /,
        *,
        identifier: str | None = None,
        track_state: bool = False,
        track_attributes: List[str] | None = None,
        field_mappings: Dict[str, str] = {},
    ) -> Callable[[C], C]: ...

    def agent(
        self,
        func: Callable | None = None,
        /,
        *,
        identifier: str | None = None,
        track_state: bool = False,
        track_attributes: List[str] | None = None,
        field_mappings: Dict[str, str] = {},
    ) -> Callable | None:
        """
        Agent decorator that creates an agent ID and propagates it to child spans.
        Also captures and propagates the class name if the decorated function is a method.
        Optionally captures instance name based on the specified identifier attribute.

        This decorator should be used in combination with @observe decorator:

        class MyAgent:
            def __init__(self, name):
                self.name = name

            @judgment.agent(identifier="name")
            @judgment.observe(span_type="function")
            def my_agent_method(self):
                # This span and all child spans will have:
                # - agent_id: auto-generated UUID
                # - class_name: "MyAgent"
                # - instance_name: self.name value
                pass

        Args:
            identifier: Name of the instance attribute to use as the instance name
        """
        if func is None:
            return partial(
                self.agent,
                identifier=identifier,
                track_state=track_state,
                track_attributes=track_attributes,
                field_mappings=field_mappings,
            )

        if not self.enable_monitoring:
            return func

        class_name = None
        if hasattr(func, "__qualname__") and "." in func.__qualname__:
            parts = func.__qualname__.split(".")
            if len(parts) >= 2:
                class_name = parts[-2]

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with async_agent_context(
                    tracer=self,
                    args=args,
                    class_name=class_name,
                    identifier=identifier,
                    track_state=track_state,
                    track_attributes=track_attributes,
                    field_mappings=field_mappings,
                ):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with sync_agent_context(
                    tracer=self,
                    args=args,
                    class_name=class_name,
                    identifier=identifier,
                    track_state=track_state,
                    track_attributes=track_attributes,
                    field_mappings=field_mappings,
                ):
                    return func(*args, **kwargs)

            return sync_wrapper

    def wrap(self, client: ApiClient) -> ApiClient:
        return wrap_provider(self, client)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all pending spans and block until completion.

        Args:
            timeout_millis: Maximum time to wait for flush completion in milliseconds

        Returns:
            True if all processors flushed successfully within timeout, False otherwise
        """
        success = True
        for processor in self.processors:
            try:
                result = processor.force_flush(timeout_millis)
                if not result:
                    success = False
            except Exception as e:
                judgeval_logger.warning(f"Error flushing processor {processor}: {e}")
                success = False
        return success

    def _atexit_flush(self) -> None:
        """Internal method called on program exit to flush remaining spans.

        This blocks until all spans are flushed or timeout is reached to ensure
        proper cleanup before program termination.
        """
        try:
            self.force_flush(timeout_millis=30000)
        except Exception as e:
            judgeval_logger.warning(f"Error during atexit flush: {e}")

    def async_evaluate(
        self,
        /,
        *,
        scorer: Union[ExampleAPIScorerConfig, BaseScorer],
        example: Example,
        model: str = JUDGMENT_DEFAULT_GPT_MODEL,
        sampling_rate: float = 1.0,
    ):
        if not self.enable_evaluation or not self.enable_monitoring:
            judgeval_logger.info("Evaluation is not enabled, skipping evaluation")
            return

        if not isinstance(scorer, (ExampleAPIScorerConfig, BaseScorer)):
            judgeval_logger.error(
                "Scorer must be an instance of ExampleAPIScorerConfig or BaseScorer, got %s, skipping evaluation."
                % type(scorer)
            )
            return

        if not isinstance(example, Example):
            judgeval_logger.error(
                "Example must be an instance of Example, got %s, skipping evaluation."
                % type(example)
            )
            return

        if sampling_rate < 0 or sampling_rate > 1:
            judgeval_logger.error(
                "Sampling rate must be between 0 and 1, got %s, skipping evaluation."
                % sampling_rate
            )
            return

        percentage = random.uniform(0, 1)
        if percentage > sampling_rate:
            judgeval_logger.info(
                "Sampling rate is %s, skipping evaluation." % sampling_rate
            )
            return

        span_context = self.get_current_span().get_span_context()
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")
        hosted_scoring = isinstance(scorer, ExampleAPIScorerConfig) or (
            isinstance(scorer, BaseScorer) and scorer.server_hosted
        )
        eval_run_name = f"async_evaluate_{span_id}"  # note this name doesnt matter because we don't save the experiment only the example and scorer_data
        if hosted_scoring:
            eval_run = ExampleEvaluationRun(
                organization_id=self.organization_id,
                project_name=self.project_name,
                eval_name=eval_run_name,
                examples=[example],
                scorers=[scorer],
                model=model,
                trace_span_id=span_id,
                trace_id=trace_id,
            )
            self.api_client.add_to_run_eval_queue_examples(
                eval_run.model_dump(warnings=False)
            )  # type: ignore
        else:
            # Handle custom scorers using local evaluation queue
            eval_run = ExampleEvaluationRun(
                organization_id=self.organization_id,
                project_name=self.project_name,
                eval_name=eval_run_name,
                examples=[example],
                scorers=[scorer],
                model=model,
                trace_span_id=span_id,
                trace_id=trace_id,
            )

            # Enqueue the evaluation run to the local evaluation queue
            self.local_eval_queue.enqueue(eval_run)

    def wait_for_completion(self, timeout: Optional[float] = 30.0) -> bool:
        """Wait for all evaluations and span processing to complete.

        This method blocks until all queued evaluations are processed and
        all pending spans are flushed to the server.

        Args:
            timeout: Maximum time to wait in seconds. Defaults to 30 seconds.
                    None means wait indefinitely.

        Returns:
            True if all processing completed within the timeout, False otherwise.

        """
        try:
            judgeval_logger.debug(
                "Waiting for all evaluations and spans to complete..."
            )

            # Wait for all queued evaluation work to complete
            eval_completed = self.local_eval_queue.wait_for_completion()
            if not eval_completed:
                judgeval_logger.warning(
                    f"Local evaluation queue did not complete within {timeout} seconds"
                )
                return False

            self.force_flush()

            judgeval_logger.debug("All evaluations and spans completed successfully")
            return True

        except Exception as e:
            judgeval_logger.warning(f"Error while waiting for completion: {e}")
            return False


def wrap(client: ApiClient) -> ApiClient:
    if not Tracer._active_tracers:
        warn(
            "No active tracers found, client will not be wrapped. "
            "You can use the global `wrap` function after creating a tracer instance. "
            "Or you can use the `wrap` method on the tracer instance to directly wrap the client. ",
            JudgmentWarning,
            stacklevel=2,
        )

    wrapped_client = client
    for tracer in Tracer._active_tracers:
        wrapped_client = tracer.wrap(wrapped_client)
    return wrapped_client


def format_inputs(
    f: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        params = list(inspect.signature(f).parameters.values())
        inputs = {}
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


# Export processor classes for direct access
__all__ = [
    "Tracer",
    "wrap",
    "JudgmentSpanProcessor",
    "NoOpJudgmentSpanProcessor",
    "NoOpSpanProcessor",
]
