from judgeval.logger import judgeval_logger


from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")


def dont_throw(func: Callable[..., T]) -> Callable[..., T | None]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            judgeval_logger.debug(
                f"An exception was raised in {func.__name__}", exc_info=e
            )
            pass

    return wrapper
