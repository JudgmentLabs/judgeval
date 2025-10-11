from functools import wraps
from typing import (
    Callable,
    TypeVar,
    Any,
    Dict,
    Mapping,
    ParamSpec,
    TypeAlias,
    AsyncGenerator,
    cast,
)

from judgeval.utils.decorators.dont_throw import dont_throw

P = ParamSpec("P")
Y = TypeVar("Y")
S = TypeVar("S")
Ctx: TypeAlias = Dict[str, Any]
ImmCtx: TypeAlias = Mapping[str, Any]


def _void_pre_hook(ctx: Ctx) -> None:
    pass


def _void_yield_hook(ctx: ImmCtx, value: Any) -> None:
    pass


def _void_post_hook(ctx: ImmCtx) -> None:
    pass


def _void_error_hook(ctx: ImmCtx, error: Exception) -> None:
    pass


def _void_finally_hook(ctx: ImmCtx) -> None:
    pass


def immutable_wrap_async_generator(
    func: Callable[P, AsyncGenerator[Y, S]],
    /,
    *,
    pre_hook: Callable[[Ctx], None] = _void_pre_hook,
    yield_hook: Callable[[ImmCtx, Y], None] = _void_yield_hook,
    post_hook: Callable[[ImmCtx], None] = _void_post_hook,
    error_hook: Callable[[ImmCtx, Exception], None] = _void_error_hook,
    finally_hook: Callable[[ImmCtx], None] = _void_finally_hook,
) -> Callable[P, AsyncGenerator[Y, S]]:
    """
    Wraps an async generator function with lifecycle hooks. Hooks MUST NOT mutate yielded values.

    - pre_hook: called when generator function is invoked (before first yield)
    - yield_hook: called after each yield with readonly context and yielded value
    - post_hook: called when generator completes successfully with readonly context
    - error_hook: called if generator raises an exception with readonly context and error
    - finally_hook: called when generator closes, always executes

    The wrapped generator yields values unchanged, and exceptions are re-raised.
    """

    pre_hook = dont_throw(pre_hook)
    yield_hook = dont_throw(yield_hook)
    post_hook = dont_throw(post_hook)
    error_hook = dont_throw(error_hook)
    finally_hook = dont_throw(finally_hook)

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[Y, S]:
        ctx: Ctx = {}
        pre_hook(ctx)
        try:
            gen = func(*args, **kwargs)
            sent_value = cast(S, None)
            while True:
                try:
                    value = await gen.asend(sent_value)
                    yield_hook(ctx, value)
                    sent_value = yield value
                except StopAsyncIteration:
                    post_hook(ctx)
                    return
        except Exception as e:
            error_hook(ctx, e)
            raise
        finally:
            finally_hook(ctx)

    return wrapper
