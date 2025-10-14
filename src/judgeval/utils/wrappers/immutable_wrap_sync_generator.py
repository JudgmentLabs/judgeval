from functools import wraps

from typing import (
    Callable,
    TypeVar,
    Any,
    Dict,
    ParamSpec,
    Generator,
    cast,
    Concatenate,
)

from judgeval.utils.decorators.dont_throw import dont_throw

P = ParamSpec("P")
Y = TypeVar("Y")
S = TypeVar("S")
R = TypeVar("R")
Ctx = Dict[str, Any]


def _void_pre_hook(ctx: Ctx, *args: Any, **kwargs: Any) -> None:
    pass


def _void_yield_hook(ctx: Ctx, value: Any) -> None:
    pass


def _void_post_hook(ctx: Ctx, result: Any) -> None:
    pass


def _void_error_hook(ctx: Ctx, error: Exception) -> None:
    pass


def _void_finally_hook(ctx: Ctx) -> None:
    pass


def immutable_wrap_sync_generator(
    func: Callable[P, Generator[Y, S, R]],
    /,
    *,
    pre_hook: Callable[Concatenate[Ctx, P], None] = _void_pre_hook,
    yield_hook: Callable[[Ctx, Y], None] = _void_yield_hook,
    post_hook: Callable[[Ctx, R], None] = _void_post_hook,
    error_hook: Callable[[Ctx, Exception], None] = _void_error_hook,
    finally_hook: Callable[[Ctx], None] = _void_finally_hook,
) -> Callable[P, Generator[Y, S, R]]:
    """
    Wraps a generator function with lifecycle hooks.

    - pre_hook: called when generator function is invoked with (ctx, *args, **kwargs) matching func's signature
    - yield_hook: called after each yield with (ctx, yielded_value)
    - post_hook: called when generator completes successfully with (ctx, return_value)
    - error_hook: called if generator raises an exception with (ctx, error)
    - finally_hook: called when generator closes with (ctx)

    The wrapped generator yields values unchanged, and exceptions are re-raised.
    """

    pre_hook = dont_throw(pre_hook)
    yield_hook = dont_throw(yield_hook)
    post_hook = dont_throw(post_hook)
    error_hook = dont_throw(error_hook)
    finally_hook = dont_throw(finally_hook)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[Y, S, R]:
        ctx: Ctx = {}
        pre_hook(ctx, *args, **kwargs)
        try:
            gen = func(*args, **kwargs)
            sent_value = cast(S, None)
            while True:
                try:
                    value = gen.send(sent_value)
                    yield_hook(ctx, value)
                    sent_value = yield value
                except StopIteration as e:
                    result = e.value
                    post_hook(ctx, result)
                    return result
        except Exception as e:
            error_hook(ctx, e)
            raise
        finally:
            finally_hook(ctx)

    return wrapper
