from __future__ import annotations

import collections
import os
import threading
import weakref

from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    set_value,
)

from judgeval.conversations.observation_exporter import ObservationExporter
from judgeval.internal.api.models import IngestObservationsRequestObservation
from judgeval.logger import judgeval_logger

# The most observations the server accepts in one request.
MAX_EXPORT_BATCH_SIZE = 100
# The span processors' defaults (OpenTelemetry's), so observations and spans
# are held and sent alike.
MAX_QUEUE_SIZE = 2048
SCHEDULE_DELAY_SECONDS = 5.0


class BatchObservationProcessor:
    """Queues observations and exports them in batches from a background
    thread.

    Modeled on OpenTelemetry's ``BatchProcessor``, which batches spans and
    logs: a full queue drops its oldest observation, the worker exports a
    batch every ``SCHEDULE_DELAY_SECONDS`` or as soon as a batch fills, and a
    forked child process starts with its own empty queue and worker.
    """

    def __init__(self, exporter: ObservationExporter):
        self._exporter = exporter
        self._shutdown = False
        self._dropped_reported = False
        self._start()
        if hasattr(os, "register_at_fork"):
            reinit = weakref.WeakMethod(self._start)

            def reinit_in_child() -> None:
                start = reinit()
                if start is not None:
                    start()

            os.register_at_fork(after_in_child=reinit_in_child)

    def _start(self) -> None:
        # Also runs in a forked child, where the parent's worker thread does
        # not exist and its queued observations are the parent's to send.
        self._queue: collections.deque[IngestObservationsRequestObservation] = (
            collections.deque(maxlen=MAX_QUEUE_SIZE)
        )
        self._export_lock = threading.Lock()
        self._worker_awaken = threading.Event()
        self._worker_thread = threading.Thread(
            name="JudgevalBatchObservationProcessor", target=self._worker, daemon=True
        )
        self._worker_thread.start()

    def emit(self, observation: IngestObservationsRequestObservation) -> None:
        if self._shutdown:
            return
        if len(self._queue) == MAX_QUEUE_SIZE and not self._dropped_reported:
            self._dropped_reported = True
            judgeval_logger.warning(
                "[Conversations] Queue full, dropping the oldest observations"
            )
        # A full deque drops its oldest observation to make room.
        self._queue.append(observation)
        if (
            len(self._queue) >= MAX_EXPORT_BATCH_SIZE
            and not self._worker_awaken.is_set()
        ):
            self._worker_awaken.set()

    def force_flush(self) -> bool:
        """Exports every queued observation on the calling thread.

        Returns:
            False if the processor is shut down.
        """
        if self._shutdown:
            return False
        self._export(everything=True)
        return True

    def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Exports the queue, then stops; waits at most ``timeout_seconds``."""
        if self._shutdown:
            return
        self._shutdown = True
        self._worker_awaken.set()
        self._worker_thread.join(timeout_seconds)

    def _worker(self) -> None:
        while not self._shutdown:
            self._worker_awaken.wait(SCHEDULE_DELAY_SECONDS)
            if self._shutdown:
                break
            self._export(everything=False)
            self._worker_awaken.clear()
        self._export(everything=True)

    def _export(self, *, everything: bool) -> None:
        """Exports one batch, then more while a full batch is waiting, or
        until the queue is empty when ``everything`` is set."""
        with self._export_lock:
            exported_one = False
            while self._queue and (
                everything
                or not exported_one
                or len(self._queue) >= MAX_EXPORT_BATCH_SIZE
            ):
                exported_one = True
                batch = [
                    self._queue.popleft()
                    for _ in range(min(MAX_EXPORT_BATCH_SIZE, len(self._queue)))
                ]
                # The export's own HTTP request must not be recorded as a
                # span of the user's app.
                token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
                try:
                    self._exporter.export(batch)
                except Exception as error:
                    judgeval_logger.error(
                        f"[Conversations] Failed to export {len(batch)} "
                        f"observations: {error!r}"
                    )
                finally:
                    detach(token)
