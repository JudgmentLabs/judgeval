from typing import Callable, TypeVar, ParamSpec

from judgeval.logger import judgeval_logger

P = ParamSpec("P")
T = TypeVar("T")


def identity_on_throw(func: Callable[P, T]) -> Callable[P, T]:
    """
    Wraps a mutation function to preserve the last argument (identity) if it fails.

    This is used for mutation hooks where we want to fall back to the original value
    if the mutation fails, ensuring the wrapper is always safe and non-breaking.

    Args:
        func: A mutation function where the last positional argument is the value to mutate.
              The function should return a potentially modified version of this value.

    Returns:
        A wrapped function that returns the last positional argument (original value) if mutation fails
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            judgeval_logger.debug(
                f"[Caught] Mutation function {func.__name__} failed, using identity",
                exc_info=e,
            )
            # The last positional argument is always the value to mutate
            return args[-1]  # type: ignore[return-value]

    return wrapper
