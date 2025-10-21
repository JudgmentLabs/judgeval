from functools import wraps
from typing import Callable, TypeVar, Any, Dict, ParamSpec, Concatenate

from judgeval.utils.decorators.dont_throw import dont_throw

P = ParamSpec("P")
R = TypeVar("R")
Ctx = Dict[str, Any]


def _void_pre_hook(ctx: Ctx, *args: Any, **kwargs: Any) -> None:
    pass


def _void_post_hook(ctx: Ctx, result: Any) -> None:
    pass


def _void_error_hook(ctx: Ctx, error: Exception) -> None:
    pass


def _void_finally_hook(ctx: Ctx) -> None:
    pass


def immutable_wrap_sync(
    func: Callable[P, R],
    /,
    *,
    pre_hook: Callable[Concatenate[Ctx, P], None] = _void_pre_hook,
    post_hook: Callable[[Ctx, R], None] = _void_post_hook,
    error_hook: Callable[[Ctx, Exception], None] = _void_error_hook,
    finally_hook: Callable[[Ctx], None] = _void_finally_hook,
) -> Callable[P, R]:
    """
    Wraps a function with lifecycle hooks.

    - pre_hook: called before func with (ctx, *args, **kwargs) matching func's signature
    - post_hook: called after successful func execution with (ctx, result)
    - error_hook: called if func raises an exception with (ctx, error)
    - finally_hook: called in finally block with (ctx)

    The wrapped function's result is returned unchanged, and exceptions are re-raised.
    """

    pre_hook = dont_throw(pre_hook)
    post_hook = dont_throw(post_hook)
    error_hook = dont_throw(error_hook)
    finally_hook = dont_throw(finally_hook)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        ctx: Ctx = {}
        pre_hook(ctx, *args, **kwargs)
        try:
            result = func(*args, **kwargs)
            post_hook(ctx, result)
            return result
        except Exception as e:
            error_hook(ctx, e)
            raise
        finally:
            finally_hook(ctx)

    return wrapper
