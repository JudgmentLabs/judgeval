from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, ParamSpec

from judgeval.env import JUDGMENT_BACKGROUND_WORKERS
from judgeval.logger import judgeval_logger

P = ParamSpec("P")

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


def submit_background(fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs) -> None:
    def task() -> None:
        try:
            fn(*args, **kwargs)
        except Exception as e:
            judgeval_logger.error(
                f"[Caught] An exception was raised in {fn.__name__}", exc_info=e
            )

    _get_executor().submit(task)
