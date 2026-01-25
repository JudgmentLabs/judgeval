from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from judgeval.env import JUDGMENT_BACKGROUND_WORKERS
from judgeval.logger import judgeval_logger

_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=JUDGMENT_BACKGROUND_WORKERS,
            thread_name_prefix="judgeval-bg",
        )
        atexit.register(_shutdown)
    return _executor


def _shutdown() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=False)
        _executor = None


def _run_safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as e:
        judgeval_logger.error(
            f"[Caught] An exception was raised in {fn.__name__}", exc_info=e
        )


def submit_background(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    _get_executor().submit(_run_safe, fn, *args, **kwargs)
