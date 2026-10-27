from __future__ import annotations

import atexit
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable

from judgeval.env import JUDGMENT_BG_MAX_QUEUE, JUDGMENT_BG_WORKERS
from judgeval.logger import judgeval_logger


class BackgroundQueue:
    """Thread-pool backed background task queue (singleton).

    Used internally to dispatch non-blocking work such as tagging and
    evaluation payloads without blocking the caller.
    """

    _instance: BackgroundQueue | None = None
    _lock = threading.Lock()

    def __init__(self, workers: int, max_queue_size: int):
        self._max_queue_size = max_queue_size
        self._semaphore = threading.Semaphore(max_queue_size)
        self._futures: set[Future[Any]] = set()
        self._state_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="judgeval-bg",
        )
        self._shutdown = False
        atexit.register(self.shutdown)

    @classmethod
    def get_instance(cls) -> BackgroundQueue:
        """Return the singleton ``BackgroundQueue`` instance, creating it if needed."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        workers=JUDGMENT_BG_WORKERS,
                        max_queue_size=JUDGMENT_BG_MAX_QUEUE,
                    )
        return cls._instance

    def enqueue(self, fn: Callable[[], Any]) -> bool:
        """Submit a callable to run in the background.

        Args:
            fn: Zero-argument callable to execute.

        Returns:
            True if the job was accepted, False if the queue is full or shut down.
        """
        with self._state_lock:
            if self._shutdown:
                return False
            accepted = self._semaphore.acquire(blocking=False)
            if accepted:
                try:
                    future = self._executor.submit(fn)
                except BaseException:
                    self._semaphore.release()
                    raise
                self._futures.add(future)
        if not accepted:
            judgeval_logger.warning("[BackgroundQueue] Queue full, dropping job")
            return False
        # A completed future runs its callback immediately, so register it
        # outside the state lock that the callback itself acquires.
        future.add_done_callback(self._on_done)
        return True

    # --- Completion and drain helpers ---
    def _on_done(self, future: Future[Any]) -> None:
        self._semaphore.release()
        with self._state_lock:
            self._futures.discard(future)
        exc = future.exception()
        if exc is not None:
            judgeval_logger.error(f"[BackgroundQueue] Job failed: {repr(exc)}")

    def _wait_for_futures(
        self, futures: tuple[Future[Any], ...], timeout_ms: int
    ) -> bool:
        """Drain a stable snapshot without blocking admission or callbacks."""
        _, not_done = wait(futures, timeout=timeout_ms / 1000.0)
        if not_done:
            judgeval_logger.warning(
                f"[BackgroundQueue] Flush timed out, {len(not_done)} jobs still pending"
            )
            return False
        return True

    # --- End completion and drain helpers ---

    def force_flush(self, timeout_ms: int = 30000) -> bool:
        """Wait for all pending jobs to complete.

        Args:
            timeout_ms: Maximum time in milliseconds to wait.

        Returns:
            True if all jobs completed within the timeout.
        """
        with self._state_lock:
            if self._shutdown:
                return False
            futures = tuple(self._futures)
        return self._wait_for_futures(futures, timeout_ms)

    def shutdown(self, timeout_ms: int = 30000) -> None:
        """Flush pending jobs and shut down the thread pool.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for flush.
        """
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
            futures = tuple(self._futures)
        try:
            self._wait_for_futures(futures, timeout_ms)
        finally:
            self._executor.shutdown(wait=False)


def enqueue(fn: Callable[[], Any]) -> bool:
    """Submit a callable to the global background queue.

    Args:
        fn: Zero-argument callable to execute.

    Returns:
        True if the job was accepted.
    """
    return BackgroundQueue.get_instance().enqueue(fn)


def flush(timeout_ms: int = 30000) -> bool:
    """Flush the global background queue.

    Args:
        timeout_ms: Maximum time in milliseconds to wait.

    Returns:
        True if all jobs completed within the timeout.
    """
    return BackgroundQueue.get_instance().force_flush(timeout_ms)


__all__ = ["BackgroundQueue", "enqueue", "flush"]
