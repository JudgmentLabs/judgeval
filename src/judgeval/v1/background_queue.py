from __future__ import annotations

import atexit
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable

from judgeval.env import JUDGMENT_BG_MAX_QUEUE, JUDGMENT_BG_WORKERS
from judgeval.logger import judgeval_logger


class BackgroundQueue:
    _instance: BackgroundQueue | None = None
    _lock = threading.Lock()

    def __init__(self, workers: int, max_queue_size: int):
        self._max_queue_size = max_queue_size
        self._semaphore = threading.Semaphore(max_queue_size)
        self._futures: set[Future[Any]] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="judgeval-bg",
        )
        self._shutdown = False
        atexit.register(self.shutdown)

    @classmethod
    def get_instance(cls) -> BackgroundQueue:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        workers=JUDGMENT_BG_WORKERS,
                        max_queue_size=JUDGMENT_BG_MAX_QUEUE,
                    )
        return cls._instance

    def enqueue(self, fn: Callable[[], Any]) -> bool:
        if self._shutdown:
            return False
        if not self._semaphore.acquire(blocking=False):
            judgeval_logger.warning("[BackgroundQueue] Queue full, dropping job")
            return False
        future = self._executor.submit(fn)
        self._futures.add(future)
        future.add_done_callback(self._on_done)
        return True

    def _on_done(self, future: Future[Any]) -> None:
        self._semaphore.release()
        self._futures.discard(future)
        exc = future.exception()
        if exc is not None:
            judgeval_logger.error(f"[BackgroundQueue] Job failed: {repr(exc)}")

    def force_flush(self, timeout_ms: int = 30000) -> bool:
        if self._shutdown:
            return False
        if not self._futures:
            return True
        _, not_done = wait(self._futures, timeout=timeout_ms / 1000.0)
        if not_done:
            judgeval_logger.warning(
                f"[BackgroundQueue] Flush timed out, {len(not_done)} jobs still pending"
            )
            return False
        return True

    def shutdown(self, timeout_ms: int = 30000) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.force_flush(timeout_ms)
        self._executor.shutdown(wait=False)


def enqueue(fn: Callable[[], Any]) -> bool:
    return BackgroundQueue.get_instance().enqueue(fn)


def flush(timeout_ms: int = 30000) -> bool:
    return BackgroundQueue.get_instance().force_flush(timeout_ms)


__all__ = ["BackgroundQueue", "enqueue", "flush"]
