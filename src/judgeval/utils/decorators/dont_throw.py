from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, overload

from judgeval.logger import judgeval_logger

T = TypeVar("T")
D = TypeVar("D")
P = ParamSpec("P")


@overload
def dont_throw(func: Callable[P, T], /) -> Callable[P, T | None]: ...


@overload
def dont_throw(
    func: None = None, /, *, default: D
) -> Callable[[Callable[P, T]], Callable[P, T | D]]: ...


def dont_throw(func: Callable[P, T] | None = None, /, *, default: Any = None):
    def decorator(f: Callable[P, T]) -> Callable[P, T | Any]:
        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | Any:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                judgeval_logger.debug(
                    f"[Caught] An exception was raised in {f.__name__}", exc_info=e
                )
                return default

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
