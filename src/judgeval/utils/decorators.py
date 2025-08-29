from functools import lru_cache, wraps
from typing import Callable, TypeVar

T = TypeVar("T")


def use_once(func: Callable[..., T]) -> Callable[..., T]:
    @lru_cache(maxsize=1)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def dont_throw(func: Callable[..., T]) -> Callable[..., T | None]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass

    return wrapper
