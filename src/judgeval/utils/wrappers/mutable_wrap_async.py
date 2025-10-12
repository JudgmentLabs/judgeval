from functools import wraps
from typing import (
    Awaitable,
    Callable,
    TypeVar,
    Any,
    Dict,
    Mapping,
    ParamSpec,
    Concatenate,
    cast,
)

from judgeval.utils.decorators.dont_throw import dont_throw

P = ParamSpec("P")
R = TypeVar("R")
Ctx = Dict[str, Any]
ImmCtx = Mapping[str, Any]


def _void_pre_hook(ctx: Ctx, *args: Any, **kwargs: Any) -> None:
    pass


def _identity_args(ctx: Ctx, args: tuple[Any, ...]) -> tuple[Any, ...]:
    return args


def _identity_kwargs(ctx: Ctx, kwargs: dict[str, Any]) -> dict[str, Any]:
    return kwargs


def _void_post_hook(ctx: Ctx, result: Any) -> None:
    pass


def _identity_mutate_hook(ctx: Ctx, result: R) -> R:
    return result


def _void_error_hook(ctx: Ctx, error: Exception) -> None:
    pass


def _void_finally_hook(ctx: Ctx) -> None:
    pass


def mutable_wrap_async(
    func: Callable[P, Awaitable[R]],
    /,
    *,
    pre_hook: Callable[Concatenate[Ctx, P], None] = _void_pre_hook,
    mutate_args_hook: Callable[
        [Ctx, tuple[Any, ...]], tuple[Any, ...]
    ] = _identity_args,
    mutate_kwargs_hook: Callable[
        [Ctx, dict[str, Any]], dict[str, Any]
    ] = _identity_kwargs,
    post_hook: Callable[[Ctx, R], None] = _void_post_hook,
    mutate_hook: Callable[[Ctx, R], R] = _identity_mutate_hook,
    error_hook: Callable[[Ctx, Exception], None] = _void_error_hook,
    finally_hook: Callable[[Ctx], None] = _void_finally_hook,
) -> Callable[P, Awaitable[R]]:
    """
    Wraps an async function with lifecycle hooks that can mutate args, kwargs, and result.

    - pre_hook: called before func with (ctx, *args, **kwargs) matching func's signature
    - mutate_args_hook: called after pre_hook with (ctx, args), returns potentially modified args
    - mutate_kwargs_hook: called after pre_hook with (ctx, kwargs), returns potentially modified kwargs
    - post_hook: called after successful func execution with (ctx, result)
    - mutate_hook: called after post_hook with (ctx, result), returns potentially modified result
    - error_hook: called if func raises an exception with (ctx, error)
    - finally_hook: called in finally block with (ctx)

    The mutate hooks can transform args/kwargs/result. Exceptions are re-raised.
    """

    pre_hook = dont_throw(pre_hook)
    mutate_args_hook = cast(
        Callable[[Ctx, tuple[Any, ...]], tuple[Any, ...]],
        dont_throw(default=_identity_args)(mutate_args_hook),
    )
    mutate_kwargs_hook = cast(
        Callable[[Ctx, dict[str, Any]], dict[str, Any]],
        dont_throw(default=_identity_kwargs)(mutate_kwargs_hook),
    )
    post_hook = dont_throw(post_hook)
    error_hook = dont_throw(error_hook)
    finally_hook = dont_throw(finally_hook)

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        ctx: Ctx = {}
        pre_hook(ctx, *args, **kwargs)

        final_args = mutate_args_hook(ctx, args)
        final_kwargs = mutate_kwargs_hook(ctx, kwargs)

        try:
            result = await func(*final_args, **final_kwargs)
            post_hook(ctx, result)
            try:
                return mutate_hook(ctx, result)
            except Exception:
                return result
        except Exception as e:
            error_hook(ctx, e)
            raise
        finally:
            finally_hook(ctx)

    return wrapper
