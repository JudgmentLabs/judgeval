from __future__ import annotations
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
    Generator,
    AsyncGenerator,
    Iterable,
)
import contextvars
import asyncio
from functools import partial
from warnings import warn

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import (
    Status,
    StatusCode,
    Tracer as ABCTracer,
    Span,
    get_current_span,
    get_tracer_provider,
    set_tracer_provider,
    INVALID_SPAN_CONTEXT,
)

from judgeval.data.evaluation_run import ExampleEvaluationRun, TraceEvaluationRun
from judgeval.data.example import Example
from judgeval.env import (
    JUDGMENT_API_KEY,
    JUDGMENT_DEFAULT_GPT_MODEL,
    JUDGMENT_ORG_ID,
    JUDGMENT_ENABLE_MONITORING,
    JUDGMENT_ENABLE_EVALUATIONS,
)
from judgeval.logger import judgeval_logger
from judgeval.scorers.api_scorer import TraceAPIScorerConfig, ExampleAPIScorerConfig
from judgeval.scorers.example_scorer import ExampleScorer
from judgeval.tracer.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
from judgeval.tracer.managers import (
    sync_span_context,
    async_span_context,
    sync_agent_context,
    async_agent_context,
)
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.utils.guards import expect_api_key, expect_organization_id
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.meta import SingletonMeta
from judgeval.version import get_version
from judgeval.warnings import JudgmentWarning

from judgeval.tracer.keys import AttributeKeys, InternalAttributeKeys
from judgeval.api import JudgmentSyncClient
from judgeval.tracer.llm import wrap_provider
from judgeval.utils.url import url_for
from judgeval.tracer.processors import (
    JudgmentSpanProcessor,
    NoOpJudgmentSpanProcessor,
)
from judgeval.tracer.utils import set_span_attribute, TraceScorerConfig
from judgeval.utils.project import _resolve_project_id
from opentelemetry.trace import use_span

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


class Tracer(metaclass=SingletonMeta):
    __slots__ = (
        "api_key",
        "organization_id",
        "project_name",
        "enable_monitoring",
        "enable_evaluation",
        "resource_attributes",
        "api_client",
        "judgment_processor",
        "tracer",
        "agent_context",
        "customer_id",
        "_initialized",
    )

    api_key: str | None
    organization_id: str | None
    project_name: str
    enable_monitoring: bool
    enable_evaluation: bool
    resource_attributes: Optional[Dict[str, Any]]
    api_client: JudgmentSyncClient
    judgment_processor: JudgmentSpanProcessor
    tracer: ABCTracer
    agent_context: ContextVar[Optional[AgentContext]]
    customer_id: ContextVar[Optional[str]]
    _initialized: bool

    def __init__(
        self,
        /,
        *,
        project_name: str,
        api_key: str | None = None,
        organization_id: str | None = None,
        enable_monitoring: bool = JUDGMENT_ENABLE_MONITORING.lower() == "true",
        enable_evaluation: bool = JUDGMENT_ENABLE_EVALUATIONS.lower() == "true",
        resource_attributes: Optional[Dict[str, Any]] = None,
        initialize: bool = True,
    ):
        if not hasattr(self, "_initialized"):
            self._initialized = False
            self.agent_context = ContextVar("current_agent_context", default=None)
            self.customer_id = ContextVar("current_customer_id", default=None)

            self.project_name = project_name
            self.api_key = expect_api_key(api_key or JUDGMENT_API_KEY)
            self.organization_id = expect_organization_id(
                organization_id or JUDGMENT_ORG_ID
            )
            self.enable_monitoring = enable_monitoring
            self.enable_evaluation = enable_evaluation
            self.resource_attributes = resource_attributes

            if self.api_key and self.organization_id:
                self.api_client = JudgmentSyncClient(
                    api_key=self.api_key, organization_id=self.organization_id
                )
            else:
                judgeval_logger.error(
                    "API Key or Organization ID is not set. You must set them in the environment variables to use the tracer."
                )

            if initialize:
                self.initialize()

    def initialize(self) -> Tracer:
        if self._initialized:
            return self

        self.judgment_processor = NoOpJudgmentSpanProcessor()
        if self.enable_monitoring:
            project_id = _resolve_project_id(
                self.project_name, self.api_key, self.organization_id
            )
            if self.api_key and self.organization_id and project_id:
                self.judgment_processor = self.get_processor(
                    tracer=self,
                    project_name=self.project_name,
                    project_id=project_id,
                    api_key=self.api_key,
                    organization_id=self.organization_id,
                    resource_attributes=self.resource_attributes,
                )

                resource = Resource.create(self.judgment_processor.resource_attributes)
                provider = TracerProvider(resource=resource)
                provider.add_span_processor(self.judgment_processor)
                set_tracer_provider(provider)
            else:
                if self.api_key and self.organization_id:
                    judgeval_logger.error(
                        f"Failed to resolve or autocreate project {self.project_name}, please create it first at https://app.judgmentlabs.ai/org/{self.organization_id}/projects. Skipping Judgment export."
                    )

        self.tracer = get_tracer_provider().get_tracer(
            JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME,
            get_version(),
        )

        self._initialized = True
        atexit.register(self._atexit_flush)
        return self

    @staticmethod
    def get_exporter(
        project_id: str,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
    ):
        from judgeval.tracer.exporters import JudgmentSpanExporter

        api_key = api_key or JUDGMENT_API_KEY
        organization_id = organization_id or JUDGMENT_ORG_ID

        if not api_key or not organization_id:
            judgeval_logger.error(
                "API Key or Organization ID is not set. You must set them in the environment variables to use the tracer."
            )
            return None

        return JudgmentSpanExporter(
            endpoint=url_for("/otel/v1/traces"),
            api_key=api_key,
            organization_id=organization_id,
            project_id=project_id,
        )

    @staticmethod
    def get_processor(
        tracer: Tracer,
        project_name: str,
        project_id: str,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        max_queue_size: int = 2**18,
        export_timeout_millis: int = 30000,
        resource_attributes: Optional[Dict[str, Any]] = None,
    ) -> JudgmentSpanProcessor:
        """Create a JudgmentSpanProcessor using the correct constructor."""
        api_key = api_key or JUDGMENT_API_KEY
        organization_id = organization_id or JUDGMENT_ORG_ID
        if not api_key or not organization_id:
            judgeval_logger.error(
                "API Key or Organization ID is not set. You must set them in the environment variables to use the tracer."
            )
            return NoOpJudgmentSpanProcessor()
        return JudgmentSpanProcessor(
            tracer,
            project_name,
            project_id,
            api_key,
            organization_id,
            max_queue_size=max_queue_size,
            export_timeout_millis=export_timeout_millis,
            resource_attributes=resource_attributes,
        )

    def get_current_span(self):
        return get_current_span()

    def get_tracer(self):
        return self.tracer

    def get_current_agent_context(self):
        return self.agent_context

    def get_current_customer_context(self):
        return self.customer_id

    def get_span_processor(self) -> JudgmentSpanProcessor:
        """Get the internal span processor of this tracer instance."""
        return self.judgment_processor

    @dont_throw
    def set_customer_id(self, customer_id: str) -> None:
        if not customer_id:
            judgeval_logger.warning("Customer ID is empty, skipping.")
            return

        span = self.get_current_span()

        if not span or not span.is_recording():
            judgeval_logger.warning(
                "No active span found. Customer ID will not be set."
            )
            return

        if self.get_current_customer_context().get():
            judgeval_logger.warning("Customer ID is already set, skipping.")
            return

        if span and span.is_recording():
            set_span_attribute(span, AttributeKeys.JUDGMENT_CUSTOMER_ID, customer_id)
            self.get_current_customer_context().set(customer_id)

            self.get_span_processor().set_internal_attribute(
                span_context=span.get_span_context(),
                key=InternalAttributeKeys.IS_CUSTOMER_CONTEXT_OWNER,
                value=True,
            )

    def _maybe_clear_customer_context(self, span: Span) -> None:
        if self.get_span_processor().get_internal_attribute(
            span_context=span.get_span_context(),
            key=InternalAttributeKeys.IS_CUSTOMER_CONTEXT_OWNER,
            default=False,
        ):
            self.get_current_customer_context().set(None)

    @dont_throw
    def _add_agent_attributes_to_span(self, span):
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

    @dont_throw
    def _record_instance_state(self, record_point: Literal["before", "after"], span):
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

    @dont_throw
    def _add_customer_id_to_span(self, span):
        customer_id = self.get_current_customer_context().get()
        if customer_id:
            set_span_attribute(span, AttributeKeys.JUDGMENT_CUSTOMER_ID, customer_id)

    @dont_throw
    def _inject_judgment_context(self, span):
        self._add_agent_attributes_to_span(span)
        self._add_customer_id_to_span(span)

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

        if scorer is None:
            judgeval_logger.error("Prompt Scorer was not found, skipping evaluation.")
            return
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
        if span_context == INVALID_SPAN_CONTEXT:
            return
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")
        eval_run_name = f"async_trace_evaluate_{span_id}"

        eval_run = TraceEvaluationRun(
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
        generator: Generator,
        main_span: Span,
        disable_generator_yield_span: bool = False,
    ):
        """Create a traced synchronous generator that wraps each yield in a span."""
        preserved_context = contextvars.copy_context()
        return _ContextPreservedSyncGeneratorWrapper(
            self,
            generator,
            preserved_context,
            main_span,
            None,
            disable_generator_yield_span,
        )

    def _create_traced_async_generator(
        self,
        async_generator: AsyncGenerator,
        main_span: Span,
        disable_generator_yield_span: bool = False,
    ):
        """Create a traced asynchronous generator that wraps each yield in a span."""
        preserved_context = contextvars.copy_context()
        return _ContextPreservedAsyncGeneratorWrapper(
            self,
            async_generator,
            preserved_context,
            main_span,
            None,
            disable_generator_yield_span,
        )

    def _wrap_sync(
        self,
        f: Callable,
        name: Optional[str],
        attributes: Optional[Dict[str, Any]],
        scorer_config: TraceScorerConfig | None = None,
        disable_generator_yield_span: bool = False,
    ):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            n = name or f.__qualname__
            with sync_span_context(self, n, attributes) as span:
                is_return_type_generator = False

                self._inject_judgment_context(span)
                self._record_instance_state("before", span)
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

                    if inspect.isgenerator(result):
                        is_return_type_generator = True
                        set_span_attribute(
                            span, AttributeKeys.JUDGMENT_OUTPUT, "<generator>"
                        )
                        self._record_instance_state("after", span)
                        return self._create_traced_sync_generator(
                            result, span, disable_generator_yield_span
                        )
                    elif inspect.isasyncgen(result):
                        is_return_type_generator = True
                        set_span_attribute(
                            span, AttributeKeys.JUDGMENT_OUTPUT, "<async_generator>"
                        )
                        self._record_instance_state("after", span)
                        return self._create_traced_async_generator(
                            result, span, disable_generator_yield_span
                        )
                    else:
                        set_span_attribute(
                            span, AttributeKeys.JUDGMENT_OUTPUT, safe_serialize(result)
                        )
                        self._record_instance_state("after", span)
                        self._maybe_clear_customer_context(span)
                        return result
                except Exception as user_exc:
                    span.record_exception(user_exc)
                    span.set_status(Status(StatusCode.ERROR, str(user_exc)))
                    self._maybe_clear_customer_context(span)
                    raise
                finally:
                    if not is_return_type_generator:
                        span.end()

        return wrapper

    def _wrap_async(
        self,
        f: Callable,
        name: Optional[str],
        attributes: Optional[Dict[str, Any]],
        scorer_config: TraceScorerConfig | None = None,
        disable_generator_yield_span: bool = False,
    ):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            n = name or f.__qualname__
            async with async_span_context(self, n, attributes) as span:
                is_return_type_generator = False
                self._inject_judgment_context(span)
                self._record_instance_state("before", span)
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
                    if inspect.isasyncgen(result):
                        is_return_type_generator = True
                        set_span_attribute(
                            span, AttributeKeys.JUDGMENT_OUTPUT, "<async_generator>"
                        )
                        self._record_instance_state("after", span)
                        return self._create_traced_async_generator(
                            result, span, disable_generator_yield_span
                        )
                    elif inspect.isgenerator(result):
                        is_return_type_generator = True
                        set_span_attribute(
                            span, AttributeKeys.JUDGMENT_OUTPUT, "<generator>"
                        )
                        self._record_instance_state("after", span)
                        return self._create_traced_sync_generator(
                            result, span, disable_generator_yield_span
                        )
                    else:
                        set_span_attribute(
                            span, AttributeKeys.JUDGMENT_OUTPUT, safe_serialize(result)
                        )
                        self._record_instance_state("after", span)
                        self._maybe_clear_customer_context(span)
                        return result
                except Exception as user_exc:
                    span.record_exception(user_exc)
                    span.set_status(Status(StatusCode.ERROR, str(user_exc)))
                    self._maybe_clear_customer_context(span)
                    raise
                finally:
                    if not is_return_type_generator:
                        span.end()

        return wrapper

    @overload
    def observe(
        self,
        func: C,
        /,
        *,
        span_type: str | None = None,
        span_name: str | None = None,
        attributes: Optional[Dict[str, Any]] = None,
        scorer_config: TraceScorerConfig | None = None,
    ) -> C: ...

    @overload
    def observe(
        self,
        func: None = None,
        /,
        *,
        span_type: str | None = None,
        span_name: str | None = None,
        attributes: Optional[Dict[str, Any]] = None,
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
        disable_generator_yield_span: bool = False,
    ) -> Callable | None:
        if func is None:
            return partial(
                self.observe,
                span_type=span_type,
                span_name=span_name,
                attributes=attributes,
                scorer_config=scorer_config,
                disable_generator_yield_span=disable_generator_yield_span,
            )

        if not self.enable_monitoring:
            return func

        # Handle functions (including generator functions) - detect generators at runtime
        name = span_name or getattr(func, "__qualname__", "function")
        func_attributes: Dict[str, Any] = {
            AttributeKeys.JUDGMENT_SPAN_KIND: span_type,
            **(attributes or {}),
        }

        if inspect.iscoroutinefunction(func):
            return self._wrap_async(
                func, name, func_attributes, scorer_config, disable_generator_yield_span
            )
        else:
            return self._wrap_sync(
                func, name, func_attributes, scorer_config, disable_generator_yield_span
            )

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
            True if processor flushed successfully within timeout, False otherwise
        """
        try:
            return self.judgment_processor.force_flush(timeout_millis)
        except Exception as e:
            judgeval_logger.warning(f"Error flushing processor: {e}")
            return False

    def _atexit_flush(self, timeout_millis: int = 30000) -> None:
        """Internal method called on program exit to flush remaining spans.

        This blocks until all spans are flushed or timeout is reached to ensure
        proper cleanup before program termination.
        """
        try:
            self.force_flush(timeout_millis=timeout_millis)
        except Exception as e:
            judgeval_logger.warning(f"Error during atexit flush: {e}")

    @dont_throw
    def async_evaluate(
        self,
        /,
        *,
        scorer: Union[ExampleAPIScorerConfig, ExampleScorer, None],
        example: Example,
        model: Optional[str] = None,
        sampling_rate: float = 1.0,
    ):
        if not self.enable_evaluation or not self.enable_monitoring:
            judgeval_logger.info("Evaluation is not enabled, skipping evaluation")
            return

        if scorer is None:
            judgeval_logger.error("Prompt Scorer was not found, skipping evaluation.")
            return

        if not isinstance(scorer, (ExampleAPIScorerConfig, ExampleScorer)):
            judgeval_logger.error(
                "Scorer must be an instance of ExampleAPIScorerConfig or ExampleScorer, got %s, skipping evaluation."
                % type(scorer)
            )
            return

        if not isinstance(example, Example):
            judgeval_logger.error(
                "Example must be an instance of Example, got %s, skipping evaluation."
                % type(example)
            )
            return

        if model is None:
            if scorer.model is None:
                model = JUDGMENT_DEFAULT_GPT_MODEL
            else:
                model = scorer.model

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
        if span_context == INVALID_SPAN_CONTEXT:
            judgeval_logger.warning(
                "No span context was found for async_evaluate, skipping evaluation. Please make sure to use the @observe decorator on the function you are evaluating."
            )
            return

        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")
        hosted_scoring = isinstance(scorer, ExampleAPIScorerConfig) or (
            isinstance(scorer, ExampleScorer) and scorer.server_hosted
        )
        eval_run = ExampleEvaluationRun(
            project_name=self.project_name,
            # note this name doesnt matter because we don't save the experiment only the example and scorer_data
            eval_name=f"async_evaluate_{span_id}",
            examples=[example],
            scorers=[scorer],
            model=model,
            trace_span_id=span_id,
            trace_id=trace_id,
        )
        if hosted_scoring:
            self.api_client.add_to_run_eval_queue_examples(
                eval_run.model_dump(warnings=False)  # type: ignore
            )
        else:
            judgeval_logger.warning(
                "The scorer provided is not hosted, skipping evaluation."
            )


def wrap(client: ApiClient) -> ApiClient:
    try:
        tracer = Tracer.get_instance()
        if tracer is None or not isinstance(tracer, Tracer):
            warn(
                "No Tracer instance found, client will not be wrapped. "
                "Create a Tracer instance first.",
                JudgmentWarning,
                stacklevel=2,
            )
            return client
        if not tracer._initialized:
            warn(
                "Tracer not initialized, client will not be wrapped. "
                "Call Tracer.initialize() first to setup the tracer.",
                JudgmentWarning,
                stacklevel=2,
            )
            return client
        return tracer.wrap(client)
    except Exception:
        warn(
            "Error accessing tracer singleton, client will not be wrapped.",
            JudgmentWarning,
            stacklevel=2,
        )
        return client


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


class _ContextPreservedSyncGeneratorWrapper:
    """Sync generator wrapper that ensures each iteration runs in preserved context."""

    def __init__(
        self,
        tracer: Tracer,
        generator: Generator,
        context: contextvars.Context,
        span: Span,
        transform_fn: Optional[Callable[[Iterable], str]],
        disable_generator_yield_span: bool = False,
    ) -> None:
        self.tracer = tracer
        self.generator = generator
        self.context = context
        self.span = span
        self.transform_fn = transform_fn
        self._finished = False
        self.disable_generator_yield_span = disable_generator_yield_span

    def __iter__(self) -> "_ContextPreservedSyncGeneratorWrapper":
        return self

    def __next__(self) -> Any:
        try:
            # Run the generator's __next__ in the preserved context
            item = self.context.run(next, self.generator)

            if not self.disable_generator_yield_span:
                with use_span(self.span):
                    span_name = (
                        str(self.span.name)
                        if hasattr(self.span, "name")
                        else "generator_item"
                    )  # type: ignore[attr-defined]
                    with self.tracer.get_tracer().start_as_current_span(
                        span_name,
                        attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "generator_item"},
                        end_on_exit=True,
                    ) as child_span:
                        set_span_attribute(
                            child_span,
                            AttributeKeys.JUDGMENT_OUTPUT,
                            safe_serialize(item),
                        )

            return item

        except StopIteration:
            # Handle output and span cleanup when generator is exhausted
            if not self._finished:
                set_span_attribute(
                    self.span, AttributeKeys.JUDGMENT_SPAN_KIND, "generator"
                )
                self.span.end()
                self._finished = True

            raise  # Re-raise StopIteration

        except Exception as e:
            if not self._finished:
                self.span.record_exception(e)
                self.span.set_status(
                    Status(StatusCode.ERROR, str(e) or type(e).__name__)
                )
                self.tracer._maybe_clear_customer_context(self.span)
                self.span.end()
                self._finished = True

            raise

    def close(self) -> None:
        """Close the generator (minimal implementation)."""
        try:
            self.generator.close()
        finally:
            if not self._finished:
                set_span_attribute(
                    self.span, AttributeKeys.JUDGMENT_SPAN_KIND, "generator"
                )
                self.tracer._maybe_clear_customer_context(self.span)
                self.span.end()
                self._finished = True


class _ContextPreservedAsyncGeneratorWrapper:
    """Async generator wrapper that ensures each iteration runs in preserved context."""

    def __init__(
        self,
        tracer: Tracer,
        generator: AsyncGenerator,
        context: contextvars.Context,
        span: Span,
        transform_fn: Optional[Callable[[Iterable], str]],
        disable_generator_yield_span: bool = False,
    ) -> None:
        self.tracer = tracer
        self.generator = generator
        self.context = context
        self.span = span
        self.transform_fn = transform_fn
        self._finished = False
        self.disable_generator_yield_span = disable_generator_yield_span

    def __aiter__(self) -> "_ContextPreservedAsyncGeneratorWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            # Run the generator's __anext__ in the preserved context
            try:
                # Python 3.10+ approach with context parameter
                item = await asyncio.create_task(
                    self.generator.__anext__(),  # type: ignore
                    context=self.context,
                )  # type: ignore
            except TypeError:
                # Python < 3.10 fallback - context parameter not supported
                item = await self.generator.__anext__()

            if not self.disable_generator_yield_span:
                with use_span(self.span):
                    span_name = (
                        str(self.span.name)
                        if hasattr(self.span, "name")
                        else "generator_item"
                    )  # type: ignore[attr-defined]
                    with self.tracer.get_tracer().start_as_current_span(
                        span_name,
                        attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "generator_item"},
                        end_on_exit=True,
                    ) as child_span:
                        set_span_attribute(
                            child_span,
                            AttributeKeys.JUDGMENT_OUTPUT,
                            safe_serialize(item),
                        )

            return item

        except StopAsyncIteration:
            # Handle output and span cleanup when generator is exhausted
            if not self._finished:
                set_span_attribute(
                    self.span, AttributeKeys.JUDGMENT_SPAN_KIND, "generator"
                )
                self.span.end()
                self._finished = True
            raise  # Re-raise StopAsyncIteration
        except Exception as e:
            if not self._finished:
                self.span.record_exception(e)
                self.span.set_status(
                    Status(StatusCode.ERROR, str(e) or type(e).__name__)
                )
                self.tracer._maybe_clear_customer_context(self.span)
                self.span.end()
                self._finished = True

            raise

    async def aclose(self) -> None:
        """Close the async generator (minimal implementation)."""
        try:
            await self.generator.aclose()
        finally:
            if not self._finished:
                set_span_attribute(
                    self.span, AttributeKeys.JUDGMENT_SPAN_KIND, "generator"
                )
                self.tracer._maybe_clear_customer_context(self.span)
                self.span.end()
                self._finished = True


__all__ = [
    "Tracer",
    "wrap",
]
