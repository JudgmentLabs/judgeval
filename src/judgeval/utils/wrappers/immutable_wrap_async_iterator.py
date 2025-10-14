from functools import wraps
from typing import (
    Callable,
    TypeVar,
    Any,
    Dict,
    Mapping,
    ParamSpec,
    AsyncIterator,
    Concatenate,
)

from judgeval.utils.decorators.dont_throw import dont_throw

P = ParamSpec("P")
Y = TypeVar("Y")
Ctx = Dict[str, Any]
ImmCtx = Mapping[str, Any]


def _void_pre_hook(ctx: Ctx, *args: Any, **kwargs: Any) -> None:
    pass


def _void_yield_hook(ctx: Ctx, value: Any) -> None:
    pass


def _void_post_hook(ctx: Ctx) -> None:
    pass


def _void_error_hook(ctx: Ctx, error: Exception) -> None:
    pass


def _void_finally_hook(ctx: Ctx) -> None:
    pass


def immutable_wrap_async_iterator(
    func: Callable[P, AsyncIterator[Y]],
    /,
    *,
    pre_hook: Callable[Concatenate[Ctx, P], None] = _void_pre_hook,
    yield_hook: Callable[[Ctx, Y], None] = _void_yield_hook,
    post_hook: Callable[[Ctx], None] = _void_post_hook,
    error_hook: Callable[[Ctx, Exception], None] = _void_error_hook,
    finally_hook: Callable[[Ctx], None] = _void_finally_hook,
) -> Callable[P, AsyncIterator[Y]]:
    """
    Wraps an async iterator function with lifecycle hooks.

    - pre_hook: called when iterator function is invoked with (ctx, *args, **kwargs) matching func's signature
    - yield_hook: called after each yield with (ctx, yielded_value)
    - post_hook: called when iterator completes successfully with (ctx)
    - error_hook: called if iterator raises an exception with (ctx, error)
    - finally_hook: called when iterator closes with (ctx)

    The wrapped iterator yields values unchanged, and exceptions are re-raised.
    """

    pre_hook = dont_throw(pre_hook)
    yield_hook = dont_throw(yield_hook)
    post_hook = dont_throw(post_hook)
    error_hook = dont_throw(error_hook)
    finally_hook = dont_throw(finally_hook)

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[Y]:
        ctx: Ctx = {}
        pre_hook(ctx, *args, **kwargs)
        try:
            async for value in func(*args, **kwargs):
                yield_hook(ctx, value)
                yield value
            post_hook(ctx)
        except Exception as e:
            error_hook(ctx, e)
            raise
        finally:
            finally_hook(ctx)

    return wrapper
