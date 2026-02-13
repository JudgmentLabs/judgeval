from functools import wraps
import time
from typing import Callable, ParamSpec, TypeVar

from judgeval.logger import judgeval_logger

T = TypeVar("T")
P = ParamSpec("P")


def debug_time(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        judgeval_logger.debug(
            f"[DebugTime] {func.__name__} took {time.perf_counter() - start:.6f}s"
        )
        return result

    return wrapper
