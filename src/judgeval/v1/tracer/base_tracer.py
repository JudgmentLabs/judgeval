from __future__ import annotations

import functools
import inspect
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, overload

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span, SpanContext, Status, StatusCode

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import (
    ExampleEvaluationRun,
    ResolveProjectNameRequest,
    ResolveProjectNameResponse,
)
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL
from judgeval.logger import judgeval_logger
from judgeval.v1.data.example import Example
from judgeval.v1.scorers.base_scorer import BaseScorer
from judgeval.v1.tracer import attribute_keys

C = TypeVar("C", bound=Callable[..., Any])


class BaseTracer(ABC):
    __slots__ = (
        "project_name",
        "enable_evaluation",
        "api_client",
        "serializer",
        "json_encoder",
        "project_id",
    )

    TRACER_NAME = "judgeval"

    def __init__(
        self,
        project_name: str,
        enable_evaluation: bool,
        api_client: JudgmentSyncClient,
        serializer: Callable[[Any], str],
    ):
        self.project_name = project_name
        self.enable_evaluation = enable_evaluation
        self.api_client = api_client
        self.serializer = serializer
        self.json_encoder = json.dumps
        self.project_id = self._resolve_project_id(project_name)

        if self.project_id is None:
            judgeval_logger.error(
                f"Failed to resolve project {project_name}, "
                f"please create it first at https://app.judgmentlabs.ai/org/{self.api_client.organization_id}/projects. "
                "Skipping Judgment export."
            )

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def force_flush(self, timeout_millis: int) -> bool:
        pass

    @abstractmethod
    def shutdown(self, timeout_millis: int) -> None:
        pass

    def get_span_exporter(self) -> SpanExporter:
        if self.project_id is not None:
            return self._create_judgment_span_exporter(self.project_id)
        else:
            judgeval_logger.error(
                "Project not resolved; cannot create exporter, returning NoOpSpanExporter"
            )
            from judgeval.v1.tracer.exporters.noop_span_exporter import NoOpSpanExporter

            return NoOpSpanExporter()

    def get_tracer(self) -> trace.Tracer:
        return trace.get_tracer(self.TRACER_NAME)

    def set_span_kind(self, kind: str) -> None:
        if kind is None:
            return
        current_span = trace.get_current_span()
        if current_span is not None:
            current_span.set_attribute(attribute_keys.JUDGMENT_SPAN_KIND, kind)

    def set_attribute(self, key: str, value: Any) -> None:
        if not self._is_valid_key(key):
            return
        if value is None:
            return
        current_span = trace.get_current_span()
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

    def set_llm_span(self) -> None:
        self.set_span_kind("llm")

    def set_tool_span(self) -> None:
        self.set_span_kind("tool")

    def set_general_span(self) -> None:
        self.set_span_kind("span")

    def set_input(self, input_data: Any) -> None:
        self.set_attribute(attribute_keys.JUDGMENT_INPUT, input_data)

    def set_output(self, output_data: Any) -> None:
        self.set_attribute(attribute_keys.JUDGMENT_OUTPUT, output_data)

    def async_evaluate(
        self,
        scorer: BaseScorer,
        example: Example,
        model: Optional[str] = None,
    ) -> None:
        self._safe_execute(
            "evaluate scorer", lambda: self._async_evaluate_impl(scorer, example, model)
        )

    def async_trace_evaluate(
        self,
        scorer: BaseScorer,
        model: Optional[str] = None,
    ) -> None:
        self._safe_execute(
            "evaluate trace scorer",
            lambda: self._async_trace_evaluate_impl(scorer, model),
        )

    def span(self, span_name: str, callable_func: Callable[[], Any]) -> Any:
        tracer = self.get_tracer()
        with tracer.start_as_current_span(span_name) as span:
            try:
                return callable_func()
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
                raise

    @staticmethod
    def start_span(span_name: str) -> Span:
        tracer = trace.get_tracer(BaseTracer.TRACER_NAME)
        return tracer.start_span(span_name)

    def _async_evaluate_impl(
        self,
        scorer: BaseScorer,
        example: Example,
        model: Optional[str],
    ) -> None:
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
            scorer, example, model, trace_id_hex, span_id_hex
        )
        self._enqueue_evaluation(evaluation_run)

    def _async_trace_evaluate_impl(
        self,
        scorer: BaseScorer,
        model: Optional[str],
    ) -> None:
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
            scorer, model, trace_id_hex, span_id_hex
        )
        try:
            trace_eval_json = json.dumps(evaluation_run)
            current_span.set_attribute(
                attribute_keys.JUDGMENT_PENDING_TRACE_EVAL, trace_eval_json
            )
        except Exception as e:
            judgeval_logger.error(f"Failed to serialize trace evaluation: {e}")

    def _resolve_project_id(self, name: str) -> Optional[str]:
        try:
            request: ResolveProjectNameRequest = {"project_name": name}
            response: ResolveProjectNameResponse = self.api_client.projects_resolve(
                request
            )
            project_id = response.get("project_id")
            return str(project_id) if project_id is not None else None
        except Exception:
            return None

    def _build_endpoint(self, base_url: str) -> str:
        return (
            base_url + "otel/v1/traces"
            if base_url.endswith("/")
            else base_url + "/otel/v1/traces"
        )

    def _create_judgment_span_exporter(self, project_id: str) -> SpanExporter:
        from judgeval.v1.tracer.exporters.judgment_span_exporter import (
            JudgmentSpanExporter,
        )

        return JudgmentSpanExporter(
            endpoint=self._build_endpoint(self.api_client.base_url),
            api_key=self.api_client.api_key,
            organization_id=self.api_client.organization_id,
            project_id=project_id,
        )

    def _generate_run_id(self, prefix: str, span_id: Optional[str]) -> str:
        return prefix + (
            span_id if span_id is not None else str(int(time.time() * 1000))
        )

    def _create_evaluation_run(
        self,
        scorer: BaseScorer,
        example: Example,
        model: Optional[str],
        trace_id: str,
        span_id: str,
    ) -> ExampleEvaluationRun:
        run_id = self._generate_run_id("async_evaluate_", span_id)
        model_name = model if model is not None else JUDGMENT_DEFAULT_GPT_MODEL

        return ExampleEvaluationRun(
            project_name=self.project_name,
            eval_name=run_id,
            model=model_name,
            trace_id=trace_id,
            trace_span_id=span_id,
            examples=[example.to_dict()],
            judgment_scorers=[scorer.get_scorer_config()],
            custom_scorers=[],
        )

    def _create_trace_evaluation_run(
        self,
        scorer: BaseScorer,
        model: Optional[str],
        trace_id: str,
        span_id: str,
    ) -> Dict[str, Any]:
        eval_name = self._generate_run_id("async_trace_evaluate_", span_id)
        model_name = model if model is not None else JUDGMENT_DEFAULT_GPT_MODEL

        return {
            "project_name": self.project_name,
            "eval_name": eval_name,
            "model": model_name,
            "trace_id": trace_id,
            "trace_span_id": span_id,
            "judgment_scorers": [scorer.get_scorer_config()],
        }

    def _enqueue_evaluation(self, evaluation_run: ExampleEvaluationRun) -> None:
        try:
            self.api_client.add_to_run_eval_queue_examples(evaluation_run)
        except Exception as e:
            judgeval_logger.error(f"Failed to enqueue evaluation run: {e}")

    def _get_sampled_span_context(self) -> Optional[SpanContext]:
        current_span = trace.get_current_span()
        if current_span is None:
            return None
        span_context = current_span.get_span_context()
        if not span_context.is_valid or not span_context.trace_flags.sampled:
            return None
        return span_context

    def _get_sampled_span(self) -> Optional[Span]:
        current_span = trace.get_current_span()
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

    def _safe_execute(self, operation: str, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as e:
            judgeval_logger.error(f"Failed to {operation}: {e}")

    @staticmethod
    def _is_valid_key(key: str) -> bool:
        return key is not None and len(key) > 0

    @overload
    def observe(
        self,
        func: C,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> C: ...

    @overload
    def observe(
        self,
        func: None = None,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable[[C], C]: ...

    def observe(
        self,
        func: Optional[C] = None,
        span_type: Optional[str] = "span",
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> C | Callable[[C], C]:
        if func is None:
            return lambda f: self.observe(f, span_type, span_name, attributes)  # type: ignore[return-value]

        tracer = self.get_tracer()
        name = span_name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(name) as span:
                    if span_type:
                        span.set_attribute(attribute_keys.JUDGMENT_SPAN_KIND, span_type)
                    if attributes:
                        for key, value in attributes.items():
                            if value is not None:
                                serialized = (
                                    self.serializer(value)
                                    if not isinstance(value, (str, int, float, bool))
                                    else value
                                )
                                span.set_attribute(key, serialized)

                    try:
                        input_data = self._format_inputs(func, args, kwargs)
                        span.set_attribute(
                            attribute_keys.JUDGMENT_INPUT, self.serializer(input_data)
                        )

                        result = await func(*args, **kwargs)

                        span.set_attribute(
                            attribute_keys.JUDGMENT_OUTPUT, self.serializer(result)
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
                        span.set_attribute(attribute_keys.JUDGMENT_SPAN_KIND, span_type)
                    if attributes:
                        for key, value in attributes.items():
                            if value is not None:
                                serialized = (
                                    self.serializer(value)
                                    if not isinstance(value, (str, int, float, bool))
                                    else value
                                )
                                span.set_attribute(key, serialized)

                    try:
                        input_data = self._format_inputs(func, args, kwargs)
                        span.set_attribute(
                            attribute_keys.JUDGMENT_INPUT, self.serializer(input_data)
                        )

                        result = func(*args, **kwargs)

                        span.set_attribute(
                            attribute_keys.JUDGMENT_OUTPUT, self.serializer(result)
                        )
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return sync_wrapper  # type: ignore[return-value]

    def _format_inputs(
        self, f: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
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
