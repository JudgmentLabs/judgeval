"""Async utilities for judgeval."""

import asyncio
import concurrent.futures
from typing import Awaitable, TypeVar


T = TypeVar("T")


def safe_run_async(coro: Awaitable[T]) -> T:  # type: ignore[type-var]
    """Safely execute an async *coro* from synchronous code.

    This helper handles two common situations:

    1. **No running event loop** - Simply delegates to ``asyncio.run``.
    2. **Existing running loop** - Executes the coroutine in a separate
       thread so that we don't attempt to nest event loops (which would raise
       ``RuntimeError``).

    Args:
        coro: The coroutine to execute.

    Returns:
        The result returned by *coro*.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: asyncio.run(coro))
        return future.result()
