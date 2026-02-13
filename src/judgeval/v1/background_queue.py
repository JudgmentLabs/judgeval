from __future__ import annotations

import atexit
import threading
from collections import deque
from typing import Any, Callable

from judgeval.logger import judgeval_logger


class BackgroundQueue:
    _instance: BackgroundQueue | None = None
    _lock = threading.Lock()

    def __init__(self, max_queue_size: int = 1024):
        self._queue: deque[Callable[[], Any]] = deque(maxlen=max_queue_size)
        self._shutdown = False
        self._event = threading.Event()
        self._process_lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        atexit.register(self.shutdown)

    @classmethod
    def get_instance(cls) -> BackgroundQueue:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def enqueue(self, fn: Callable[[], Any]) -> bool:
        judgeval_logger.debug(f"[BackgroundQueue] Enqueuing job: {repr(fn)}")
        if self._shutdown:
            return False
        self._queue.append(fn)
        self._event.set()
        return True

    def _worker_loop(self) -> None:
        while not self._shutdown:
            self._event.wait()
            self._event.clear()
            self._process_queue()

    def _process_queue(self) -> None:
        with self._process_lock:
            while self._queue:
                fn = self._queue.popleft()
                try:
                    fn()
                except Exception as e:
                    judgeval_logger.error(f"[BackgroundQueue] Job failed: {repr(e)}")

    def force_flush(self, timeout_ms: int = 30000) -> bool:
        judgeval_logger.debug(
            f"[BackgroundQueue] Flushing queue with timeout: {timeout_ms}ms"
        )
        if self._shutdown:
            return False
        self._process_queue()
        return True

    def shutdown(self, timeout_ms: int = 5000) -> None:
        judgeval_logger.debug(
            f"[BackgroundQueue] Shutting down with timeout: {timeout_ms}ms"
        )
        if self._shutdown:
            return
        self._shutdown = True
        self._event.set()
        self._process_queue()
        self._worker.join(timeout=timeout_ms / 1000.0)


def enqueue(fn: Callable[[], Any]) -> bool:
    return BackgroundQueue.get_instance().enqueue(fn)


def flush(timeout_ms: int = 30000) -> bool:
    return BackgroundQueue.get_instance().force_flush(timeout_ms)
