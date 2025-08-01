"""
Custom OpenTelemetry span processor for Judgment API.

This processor uses BatchSpanProcessor to handle batching and export
of TraceSpan objects converted to OpenTelemetry format.
"""

from __future__ import annotations

import threading
import queue
from typing import Any, Dict, Optional, Callable

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.trace import Status, StatusCode, SpanContext, TraceFlags
from opentelemetry.trace.span import TraceState, INVALID_SPAN_CONTEXT

from judgeval.common.logger import judgeval_logger
from judgeval.common.tracer.otel_exporter import JudgmentAPISpanExporter
from judgeval.common.tracer.span_processor import SpanProcessorBase
from judgeval.common.tracer.span_transformer import SpanTransformer
from judgeval.data import TraceSpan
from judgeval.evaluation_run import EvaluationRun


class SimpleReadableSpan(ReadableSpan):
    """Simple ReadableSpan implementation that wraps TraceSpan data."""

    def __init__(self, trace_span: TraceSpan, span_state: str = "completed"):
        self._name = trace_span.function
        self._span_id = trace_span.span_id
        self._trace_id = trace_span.trace_id

        self._start_time = (
            int(trace_span.created_at * 1_000_000_000)
            if trace_span.created_at
            else None
        )
        self._end_time: Optional[int] = None

        if (
            span_state == "completed"
            and trace_span.duration is not None
            and self._start_time is not None
        ):
            self._end_time = self._start_time + int(trace_span.duration * 1_000_000_000)

        self._status = (
            Status(StatusCode.ERROR) if trace_span.error else Status(StatusCode.OK)
        )

        self._attributes: Dict[str, Any] = (
            SpanTransformer.trace_span_to_otel_attributes(trace_span, span_state)
        )

        try:
            trace_id_int = (
                int(trace_span.trace_id.replace("-", ""), 16)
                if trace_span.trace_id
                else 0
            )
            span_id_int = (
                int(trace_span.span_id.replace("-", ""), 16)
                if trace_span.span_id
                else 0
            )

            self._context = SpanContext(
                trace_id=trace_id_int,
                span_id=span_id_int,
                is_remote=False,
                trace_flags=TraceFlags(0x01),
                trace_state=TraceState(),
            )
        except (ValueError, TypeError) as e:
            judgeval_logger.warning(f"Failed to create proper SpanContext: {e}")
            self._context = INVALID_SPAN_CONTEXT

        self._parent: Optional[SpanContext] = None
        self._events: list[Any] = []
        self._links: list[Any] = []
        self._instrumentation_info: Optional[Any] = None


class JudgmentSpanProcessor(SpanProcessor, SpanProcessorBase):
    """
    Span processor that converts TraceSpan objects to OpenTelemetry format
    and uses BatchSpanProcessor for export.
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
        batch_size: int = 50,
        flush_interval: float = 1.0,
        max_queue_size: int = 2048,
        export_timeout: int = 30000,
    ):
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id

        self._span_cache: Dict[str, TraceSpan] = {}
        self._span_states: Dict[str, str] = {}
        self._cache_lock = threading.RLock()

        # Background trace upsert processing
        self._trace_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._trace_worker_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()
        self._start_trace_worker()

        self.batch_processor = BatchSpanProcessor(
            JudgmentAPISpanExporter(
                judgment_api_key=judgment_api_key,
                organization_id=organization_id,
            ),
            max_queue_size=max_queue_size,
            schedule_delay_millis=int(flush_interval * 1000),
            max_export_batch_size=batch_size,
            export_timeout_millis=export_timeout,
        )

    def _start_trace_worker(self):
        """Start the background worker thread for processing trace upserts."""
        if (
            self._trace_worker_thread is None
            or not self._trace_worker_thread.is_alive()
        ):
            self._trace_worker_thread = threading.Thread(
                target=self._trace_worker, daemon=True, name="JudgmentTraceWorker"
            )
            self._trace_worker_thread.start()

    def _trace_worker(self):
        """Background worker that processes trace upserts."""
        while not self._shutdown_flag.is_set():
            try:
                # Wait for items with a timeout to allow checking shutdown flag
                trace_item = self._trace_queue.get(timeout=1.0)

                try:
                    trace_data, upsert_callback, final_save = trace_item
                    upsert_callback(trace_data)
                    judgeval_logger.debug(
                        f"Background trace upsert completed for trace {trace_data.get('trace_id', 'unknown')}"
                    )
                except Exception as e:
                    judgeval_logger.warning(
                        f"Error processing background trace upsert: {e}"
                    )
                finally:
                    self._trace_queue.task_done()

            except queue.Empty:
                # Timeout occurred, continue loop to check shutdown flag
                continue
            except Exception as e:
                judgeval_logger.warning(f"Error in trace worker: {e}")

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self.batch_processor.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        self.batch_processor.on_end(span)

    def queue_span_update(self, span: TraceSpan, span_state: str = "input") -> None:
        if span_state == "completed":
            span.set_update_id_to_ending_number()
        else:
            span.increment_update_id()

        with self._cache_lock:
            span_id = span.span_id

            self._span_cache[span_id] = span
            self._span_states[span_id] = span_state

            self._send_span_update(span, span_state)

            if span_state == "completed" or span_state == "error":
                self._span_cache.pop(span_id, None)
                self._span_states.pop(span_id, None)

    def _send_span_update(self, span: TraceSpan, span_state: str) -> None:
        readable_span = SimpleReadableSpan(span, span_state)
        self.batch_processor.on_end(readable_span)

    def flush_pending_spans(self) -> None:
        with self._cache_lock:
            if not self._span_cache:
                return

            for span_id, span in self._span_cache.items():
                span_state = self._span_states.get(span_id, "input")
                self._send_span_update(span, span_state)

    def queue_evaluation_run(
        self, evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ) -> None:
        attributes = SpanTransformer.evaluation_run_to_otel_attributes(
            evaluation_run, span_id, span_data
        )

        readable_span = SimpleReadableSpan(span_data, "evaluation_run")
        readable_span._attributes.update(attributes)

        self.batch_processor.on_end(readable_span)

    def queue_trace_upsert(
        self,
        trace_data: Dict[str, Any],
        upsert_callback: Callable[[Dict[str, Any]], Dict[str, Any]],
        final_save: bool = False,
    ) -> None:
        """
        Queue a trace upsert to be processed in the background.

        Args:
            trace_data: The trace data to upsert
            upsert_callback: Callback function that performs the actual upsert
            final_save: Whether this is the final save
        """
        try:
            self._trace_queue.put_nowait((trace_data, upsert_callback, final_save))
            judgeval_logger.debug(
                f"Queued trace upsert for trace {trace_data.get('trace_id', 'unknown')}"
            )
        except queue.Full:
            judgeval_logger.warning(
                "Trace upsert queue is full, performing synchronous upsert"
            )
            # Fallback to synchronous processing if queue is full
            try:
                upsert_callback(trace_data)
            except Exception as e:
                judgeval_logger.error(
                    f"Error in fallback synchronous trace upsert: {e}"
                )

    def flush_pending_traces(self) -> None:
        """Flush all pending trace upserts by waiting for the queue to empty."""
        try:
            # Wait for all queued trace upserts to complete
            self._trace_queue.join()
            judgeval_logger.debug("All pending trace upserts have been processed")
        except Exception as e:
            judgeval_logger.warning(f"Error flushing pending traces: {e}")

    def shutdown(self) -> None:
        try:
            self.flush_pending_spans()
        except Exception as e:
            judgeval_logger.warning(
                f"Error flushing pending spans during shutdown: {e}"
            )

        # Shutdown trace processing
        try:
            self.flush_pending_traces()
            self._shutdown_flag.set()

            if self._trace_worker_thread and self._trace_worker_thread.is_alive():
                self._trace_worker_thread.join(timeout=5.0)

        except Exception as e:
            judgeval_logger.warning(f"Error shutting down trace worker: {e}")

        self.batch_processor.shutdown()

        with self._cache_lock:
            self._span_cache.clear()
            self._span_states.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            self.flush_pending_spans()
            self.flush_pending_traces()
        except Exception as e:
            judgeval_logger.warning(f"Error flushing pending spans and traces: {e}")

        return self.batch_processor.force_flush(timeout_millis)
