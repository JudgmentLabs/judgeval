from functools import wraps
from typing import Callable, TypeVar, Any, Dict, Mapping, ParamSpec, TypeAlias

from judgeval.utils.decorators.dont_throw import dont_throw

P = ParamSpec("P")
R = TypeVar("R")
Ctx: TypeAlias = Dict[str, Any]
ImmCtx: TypeAlias = Mapping[str, Any]


def _void_pre_hook(ctx: Ctx) -> None:
    pass


def _void_post_hook(ctx: ImmCtx, result: Any) -> None:
    pass


def _void_error_hook(ctx: ImmCtx, error: Exception) -> None:
    pass


def _void_finally_hook(ctx: ImmCtx) -> None:
    pass


def immutable_wrap_sync(
    func: Callable[P, R],
    /,
    *,
    pre_hook: Callable[[Ctx], None] = _void_pre_hook,
    post_hook: Callable[[ImmCtx, R], None] = _void_post_hook,
    error_hook: Callable[[ImmCtx, Exception], None] = _void_error_hook,
    finally_hook: Callable[[ImmCtx], None] = _void_finally_hook,
) -> Callable[P, R]:
    """
    Wraps a function with lifecycle hooks. Hooks MUST NOT mutate the result.

    - pre_hook: called before func, can populate the context dict
    - post_hook: called after successful func execution with readonly context and result
    - error_hook: called if func raises an exception with readonly context and error
    - finally_hook: called in finally block with readonly context, always executes

    The wrapped function's result is returned unchanged, and exceptions are re-raised.
    """

    pre_hook = dont_throw(pre_hook)
    post_hook = dont_throw(post_hook)
    error_hook = dont_throw(error_hook)
    finally_hook = dont_throw(finally_hook)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        ctx: Ctx = {}
        pre_hook(ctx)
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
