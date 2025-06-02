import litellm
from litellm.integrations.custom_logger import CustomLogger
from judgeval.common.tracer import TraceClient, TraceSpan,  current_span_var
from typing import Optional
import time
import uuid


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class JudgevalLitellmCallbackHandler(CustomLogger, metaclass=SingletonMeta):

    def __init__(self, tracer):
        self.tracer = tracer
        self._current_span_id: Optional[str] = None
        self._current_trace_client: Optional[TraceClient] = None

    def log_pre_api_call(self, model, messages, kwargs):
        """Start span for LLM call"""
        trace_client = self.tracer.get_current_trace()
        if not trace_client:
            return

        # Store trace client reference so it's available even if context is cleared
        self._current_trace_client = trace_client

        # Signal that a LiteLLM operation is starting
        self.tracer.litellm_operation_started()

        span_id = str(uuid.uuid4())
        self._current_span_id = span_id

        parent_span_id = current_span_var.get()
        depth = 0
        if parent_span_id and parent_span_id in trace_client.span_id_to_span:
            depth = trace_client.span_id_to_span[parent_span_id].depth + 1

        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_client.trace_id,
            parent_span_id=parent_span_id,
            function=f"LiteLLM-{model}",
            depth=depth,
            created_at=time.time(),
            span_type="llm"
        )
        span.inputs = {"model": model, "messages": messages}

        trace_client.add_span(span)

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        """Post-processing - no action needed"""
        pass

    def _finish_span(self, response_obj, start_time, end_time, error=None):
        """Complete span with results"""
        # Use stored trace client instead of context variable
        trace_client = self._current_trace_client
        if not trace_client or not self._current_span_id:
            self.tracer.litellm_operation_completed()
            return

        span = trace_client.span_id_to_span.get(self._current_span_id)
        if not span:
            self.tracer.litellm_operation_completed()
            return

        duration = None
        if start_time and end_time:
            if hasattr(start_time, 'timestamp'):
                duration = (end_time - start_time).total_seconds()
            else:
                duration = end_time - start_time

        output = error
        if not error and response_obj and hasattr(response_obj, 'choices') and response_obj.choices:
            try:
                output = response_obj.choices[0].message.content
            except:
                output = str(response_obj)

        # Update span
        span.duration = duration
        span.output = output

        # Clean up references
        self._current_span_id = None
        self._current_trace_client = None

        self.tracer.litellm_operation_completed()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Handle successful LLM call"""
        self._finish_span(response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Handle failed LLM call"""
        error = kwargs.get("exception", response_obj)
        self._finish_span(response_obj, start_time, end_time, error=error)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Handle successful async LLM call"""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Handle failed async LLM call"""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    def get_trace_client(self) -> Optional[TraceClient]:
        """Get current trace client for manual operations"""
        return self.tracer.get_current_trace()
