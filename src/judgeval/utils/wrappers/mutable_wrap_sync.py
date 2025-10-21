from functools import wraps
from typing import Callable, TypeVar, Any, Dict, ParamSpec, Concatenate

from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.utils.wrappers.utils import identity_on_throw

P = ParamSpec("P")
R = TypeVar("R")
Ctx = Dict[str, Any]


def mutable_wrap_sync(
    func: Callable[P, R],
    /,
    *,
    pre_hook: Callable[Concatenate[Ctx, P], None] | None = None,
    mutate_args_hook: Callable[[Ctx, tuple[Any, ...]], tuple[Any, ...]] | None = None,
    mutate_kwargs_hook: Callable[[Ctx, dict[str, Any]], dict[str, Any]] | None = None,
    post_hook: Callable[[Ctx, R], None] | None = None,
    mutate_hook: Callable[[Ctx, R], R] | None = None,
    error_hook: Callable[[Ctx, Exception], None] | None = None,
    finally_hook: Callable[[Ctx], None] | None = None,
) -> Callable[P, R]:
    """
    Wraps a function with lifecycle hooks that can mutate args, kwargs, and result.

    - pre_hook: called before func with (ctx, *args, **kwargs) matching func's signature
    - mutate_args_hook: called after pre_hook with (ctx, args), returns potentially modified args
    - mutate_kwargs_hook: called after pre_hook with (ctx, kwargs), returns potentially modified kwargs
    - post_hook: called after successful func execution with (ctx, result)
    - mutate_hook: called after post_hook with (ctx, result), returns potentially modified result
    - error_hook: called if func raises an exception with (ctx, error)
    - finally_hook: called in finally block with (ctx)

    The mutate hooks can transform args/kwargs/result. Exceptions are re-raised.
    """

    safe_pre_hook = dont_throw(pre_hook) if pre_hook else (lambda ctx, *a, **kw: None)
    safe_post_hook = dont_throw(post_hook) if post_hook else (lambda ctx, r: None)
    safe_error_hook = dont_throw(error_hook) if error_hook else (lambda ctx, e: None)
    safe_finally_hook = dont_throw(finally_hook) if finally_hook else (lambda ctx: None)

    safe_mutate_args = identity_on_throw(mutate_args_hook) if mutate_args_hook else None
    safe_mutate_kwargs = (
        identity_on_throw(mutate_kwargs_hook) if mutate_kwargs_hook else None
    )
    safe_mutate_hook = identity_on_throw(mutate_hook) if mutate_hook else None

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        ctx: Ctx = {}
        safe_pre_hook(ctx, *args, **kwargs)

        final_args = safe_mutate_args(ctx, args) if safe_mutate_args else args
        final_kwargs = safe_mutate_kwargs(ctx, kwargs) if safe_mutate_kwargs else kwargs

        try:
            result = func(*final_args, **final_kwargs)
            safe_post_hook(ctx, result)
            return safe_mutate_hook(ctx, result) if safe_mutate_hook else result
        except Exception as e:
            safe_error_hook(ctx, e)
            raise
        finally:
            safe_finally_hook(ctx)

    return wrapper
