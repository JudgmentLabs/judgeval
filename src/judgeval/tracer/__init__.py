from __future__ import annotations
from contextvars import ContextVar
from datetime import timedelta
import functools
import inspect
import random
from typing import (
    Any,
    Union,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    overload,
)
from functools import partial
from warnings import warn

from opentelemetry.context import Context
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import (
    Status,
    StatusCode,
    TracerProvider as ABCTracerProvider,
    NoOpTracerProvider,
    Tracer as ABCTracer,
    get_current_span,
)

from judgeval.data.example import Example
from judgeval.env import (
    JUDGMENT_API_KEY,
    JUDGMENT_API_URL,
    JUDGMENT_DEFAULT_GPT_MODEL,
    JUDGMENT_ORG_ID,
)
from judgeval.evaluation import EvaluationRun
from judgeval.exceptions import JudgmentAPIError, JudgmentRuntimeError
from judgeval.logger import judgeval_logger
from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.scorers.base_scorer import BaseScorer
from judgeval.tracer.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
from judgeval.tracer.managers import sync_span_context
from judgeval.utils.serialize import safe_serialize
from judgeval.version import get_version
from judgeval.warnings import JudgmentWarning

from judgeval.tracer.exporters import JudgmentSpanExporter
from judgeval.tracer.keys import EventKeys, ResourceKeys
from judgeval.api import JudgmentSyncClient
from judgeval.tracer.llm import wrap_provider

C = TypeVar("C", bound=Callable)
Cls = TypeVar("Cls", bound=Type)
ApiClient = TypeVar("ApiClient", bound=Any)


def resolve_project_id(
    api_key: str, organization_id: str, project_name: str
) -> str | None:
    try:
        client = JudgmentSyncClient(
            api_key=api_key,
            organization_id=organization_id,
        )
        return client.projects_resolve({"project_name": project_name})["project_id"]
    except Exception as e:
        return None


class Tracer:
    _active_tracers: MutableMapping[str, Tracer] = {}

    __slots__ = (
        "api_key",
        "organization_id",
        "project_name",
        "api_url",
        "deep_tracing",
        "enable_monitoring",
        "enable_evaluation",
        # Otel
        "processors",
        "provider",
        "tracer",
        "context",
    )

    api_key: str
    organization_id: str
    project_name: str
    api_url: str
    deep_tracing: bool
    enable_monitoring: bool
    enable_evaluation: bool

    # Judgeval supports sending raw spans to through any otel compatible span processor.
    processors: List[SpanProcessor]
    provider: ABCTracerProvider
    tracer: ABCTracer
    context: ContextVar[Context]

    def __init__(
        self,
        /,
        *,
        project_name: str,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        api_url: Optional[str] = None,
        deep_tracing: bool = False,
        enable_monitoring: bool = True,
        enable_evaluation: bool = False,
        processors: List[SpanProcessor] = [],
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
        self.api_url = api_url or JUDGMENT_API_URL

        self.deep_tracing = deep_tracing
        self.enable_monitoring = enable_monitoring
        self.enable_evaluation = enable_evaluation

        self.processors = processors
        self.provider = NoOpTracerProvider()

        # TODO:
        self.context = ContextVar(f"judgeval:tracer:{project_name}")

        project_id = resolve_project_id(
            self.api_key, self.organization_id, self.project_name
        )

        if project_id is None:
            if self.enable_monitoring:
                judgeval_logger.error(
                    f"Failed to resolve project {self.project_name}, please create it first at https://app.judgmentlabs.ai/projects. Skipping monitoring and evaluation."
                )
            self.enable_monitoring = False
            self.enable_evaluation = False

        if self.enable_monitoring:
            assert project_id is not None
            self.provider = TracerProvider(
                resource=Resource.create(
                    {
                        "service.name": self.project_name,
                        ResourceKeys.JUDGMENT_PROJECT_ID: project_id,
                    }
                ),
            )

            self.processors.append(
                BatchSpanProcessor(
                    JudgmentSpanExporter(
                        endpoint=self.api_url,
                        api_key=self.api_key,
                        organization_id=self.organization_id,
                    ),
                    max_queue_size=2**18,
                    export_timeout_millis=30000,
                )
            )

            for processor in self.processors:
                self.provider.add_span_processor(processor)

        self.tracer = self.provider.get_tracer(
            JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME,
            get_version(),
        )

        Tracer._active_tracers[self.project_name] = self

    def get_current_span(self):
        # TODO: review, need to maintain context var manually if we dont
        # want to override the default tracer provider
        return get_current_span()

    def get_tracer(self):
        return self.tracer

    def _wrap_sync(
        self, f: Callable, name: Optional[str], attributes: Optional[Dict[str, Any]]
    ):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            n = name or f.__qualname__
            with sync_span_context(self, n, attributes) as span:
                try:
                    span.add_event(
                        EventKeys.JUDGMENT_INPUT,
                        {
                            "value": safe_serialize(format_inputs(f, args, kwargs)),
                        },
                    )

                    result = f(*args, **kwargs)
                except Exception as user_exc:
                    span.record_exception(user_exc)
                    span.set_status(Status(StatusCode.ERROR, str(user_exc)))
                    raise
                if span is not None:
                    span.add_event(
                        EventKeys.JUDGMENT_OUTPUT,
                        {
                            "value": safe_serialize(result),
                        },
                    )
                return result

        return wrapper

    def _wrap_async(
        self, f: Callable, name: Optional[str], attributes: Optional[Dict[str, Any]]
    ):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            n = name or f.__qualname__
            with sync_span_context(self, n, attributes) as span:
                try:
                    span.add_event(
                        EventKeys.JUDGMENT_INPUT,
                        {
                            "value": safe_serialize(format_inputs(f, args, kwargs)),
                        },
                    )
                    result = await f(*args, **kwargs)
                except Exception as user_exc:
                    span.record_exception(user_exc)
                    span.set_status(Status(StatusCode.ERROR, str(user_exc)))
                    raise
                if span is not None:
                    span.add_event(
                        EventKeys.JUDGMENT_OUTPUT,
                        {
                            "value": safe_serialize(result),
                        },
                    )
                return result

        return wrapper

    @overload
    def observe(self, func: C, /, *, span_type: str | None = None) -> C: ...

    @overload
    def observe(
        self, func: None = None, /, *, span_type: str | None = None
    ) -> Callable[[C], C]: ...

    def observe(
        self, func: Callable | None = None, /, *, span_type: str | None = "span"
    ) -> Callable | None:
        if func is None:
            return partial(self.observe, span_type=span_type)

        if not self.enable_monitoring:
            return func

        name = func.__qualname__
        attributes = {
            "span.type": span_type,
        }

        if inspect.iscoroutinefunction(func):
            return self._wrap_async(func, name, attributes)
        else:
            return self._wrap_sync(func, name, attributes)

    @overload
    def observe_tools(
        self,
        cls: Cls,
        /,
        *,
        exclude_methods: List[str] = [],
        include_private: bool = False,
    ) -> Cls: ...

    @overload
    def observe_tools(
        self,
        cls: None = None,
        /,
        *,
        exclude_methods: List[str] = [],
        include_private: bool = False,
    ) -> Callable[[Cls], Cls]: ...

    def observe_tools(
        self,
        cls: Cls | None = None,
        /,
        *,
        exclude_methods: List[str] = [],
        include_private: bool = False,
    ) -> Cls | Callable[[Cls], Cls]:
        if cls is None:
            return partial(
                self.observe_tools,
                exclude_methods=exclude_methods,
                include_private=include_private,
            )
        return cls

    def wrap(self, client: ApiClient) -> ApiClient:
        return wrap_provider(self, client)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        for processor in self.processors:
            processor.force_flush(timeout_millis)
        return True

    def async_evaluate(
        self,
        /,
        *,
        scorer: Union[APIScorerConfig, BaseScorer],
        example: Example,
        model: str = JUDGMENT_DEFAULT_GPT_MODEL,
        sampling_rate: float = 1.0,
    ):
        if not self.enable_evaluation or not self.enable_monitoring:
            judgeval_logger.info("Evaluation is not enabled, skipping evaluation")
            return

        if not isinstance(scorer, (APIScorerConfig, BaseScorer)):
            judgeval_logger.error(
                "Scorer must be an instance of APIScorerConfig or BaseScorer, got %s, skipping evaluation."
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
    for tracer in Tracer._active_tracers.values():
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
