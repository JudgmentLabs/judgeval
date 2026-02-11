import pytest
from typing import Dict, Any, Mapping, AsyncIterator

from judgeval.utils.wrappers.immutable_wrap_async_iterator import (
    immutable_wrap_async_iterator,
)


@pytest.mark.asyncio
async def test_basic_functionality():
    """Test that wrapped async iterator executes and yields correct values."""

    async def count_to_three() -> AsyncIterator[int]:
        yield 1
        yield 2
        yield 3

    wrapped = immutable_wrap_async_iterator(count_to_three)
    result = [x async for x in wrapped()]
    assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_pre_hook_populates_context():
    """Test that pre_hook can populate context dict."""
    captured_ctx = {}

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["called"] = True
        ctx["value"] = 42

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        captured_ctx.update(ctx)

    async def simple_gen() -> AsyncIterator[str]:
        yield "result"

    wrapped = immutable_wrap_async_iterator(
        simple_gen, pre_hook=pre, finally_hook=finally_hook
    )
    [x async for x in wrapped()]

    assert captured_ctx["called"] is True
    assert captured_ctx["value"] == 42


@pytest.mark.asyncio
async def test_yield_hook_receives_each_value():
    """Test that yield_hook is called for each yielded value."""
    yielded_values = []

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        yielded_values.append(value)

    async def count() -> AsyncIterator[int]:
        yield 10
        yield 20
        yield 30

    wrapped = immutable_wrap_async_iterator(count, yield_hook=yield_hook)
    [x async for x in wrapped()]

    assert yielded_values == [10, 20, 30]


@pytest.mark.asyncio
async def test_yield_hook_reads_pre_hook_context():
    """Test that yield_hook can read what pre_hook set in context."""

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["multiplier"] = 2

    captured_contexts = []

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        captured_contexts.append(dict(ctx))

    async def gen() -> AsyncIterator[int]:
        yield 1
        yield 2

    wrapped = immutable_wrap_async_iterator(gen, pre_hook=pre, yield_hook=yield_hook)
    [x async for x in wrapped()]

    assert all(c.get("multiplier") == 2 for c in captured_contexts)


@pytest.mark.asyncio
async def test_post_hook_called_on_completion():
    """Test that post_hook is called when async iterator completes."""
    post_called = []

    def post(ctx: Mapping[str, Any]) -> None:
        post_called.append(True)

    async def simple_gen() -> AsyncIterator[int]:
        yield 1
        yield 2

    wrapped = immutable_wrap_async_iterator(simple_gen, post_hook=post)
    [x async for x in wrapped()]

    assert post_called == [True]


@pytest.mark.asyncio
async def test_post_hook_called_on_empty_iterator():
    """Test that post_hook is called even for empty async iterator."""
    post_called = []

    def post(ctx: Mapping[str, Any]) -> None:
        post_called.append(True)

    async def empty_gen() -> AsyncIterator[int]:
        return
        yield

    wrapped = immutable_wrap_async_iterator(empty_gen, post_hook=post)
    [x async for x in wrapped()]

    assert post_called == [True]


@pytest.mark.asyncio
async def test_preserves_iterator_signature():
    """Test that wrapped async iterator preserves argument types."""

    async def parameterized_gen(
        start: int, end: int, prefix: str = ""
    ) -> AsyncIterator[str]:
        for i in range(start, end):
            yield f"{prefix}{i}"

    wrapped = immutable_wrap_async_iterator(parameterized_gen)
    result = [x async for x in wrapped(1, 4, prefix="num-")]

    assert result == ["num-1", "num-2", "num-3"]


@pytest.mark.asyncio
async def test_pre_hook_exception_is_caught():
    """Test that exceptions in pre_hook are caught by dont_throw."""

    def bad_pre(ctx: Dict[str, Any]) -> None:
        raise ValueError("Pre hook error")

    async def safe_gen() -> AsyncIterator[str]:
        yield "success"

    wrapped = immutable_wrap_async_iterator(safe_gen, pre_hook=bad_pre)
    result = [x async for x in wrapped()]

    assert result == ["success"]


@pytest.mark.asyncio
async def test_yield_hook_exception_is_caught():
    """Test that exceptions in yield_hook are caught by dont_throw."""

    def bad_yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        raise RuntimeError("Yield hook error")

    async def safe_gen() -> AsyncIterator[int]:
        yield 1
        yield 2

    wrapped = immutable_wrap_async_iterator(safe_gen, yield_hook=bad_yield_hook)
    result = [x async for x in wrapped()]

    assert result == [1, 2]


@pytest.mark.asyncio
async def test_post_hook_exception_is_caught():
    """Test that exceptions in post_hook are caught by dont_throw."""

    def bad_post(ctx: Mapping[str, Any]) -> None:
        raise RuntimeError("Post hook error")

    async def safe_gen() -> AsyncIterator[int]:
        yield 42

    wrapped = immutable_wrap_async_iterator(safe_gen, post_hook=bad_post)
    result = [x async for x in wrapped()]

    assert result == [42]


@pytest.mark.asyncio
async def test_default_void_hooks():
    """Test that default void hooks work without errors."""

    async def simple() -> AsyncIterator[str]:
        yield "works"

    wrapped = immutable_wrap_async_iterator(simple)
    result = [x async for x in wrapped()]

    assert result == ["works"]


@pytest.mark.asyncio
async def test_multiple_calls_isolated_contexts():
    """Test that each async iterator call gets its own isolated context."""
    call_count = []

    def pre(ctx: Dict[str, Any], i: int) -> None:
        ctx["id"] = len(call_count)
        call_count.append(ctx["id"])

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        assert ctx["id"] == value

    async def gen(i: int) -> AsyncIterator[int]:
        yield i

    wrapped = immutable_wrap_async_iterator(gen, pre_hook=pre, yield_hook=yield_hook)

    [x async for x in wrapped(0)]
    [x async for x in wrapped(1)]
    [x async for x in wrapped(2)]

    assert call_count == [0, 1, 2]


@pytest.mark.asyncio
async def test_error_hook_called_on_exception():
    """Test that error_hook is called when async iterator raises an exception."""
    captured_error = None

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        nonlocal captured_error
        captured_error = err

    async def failing_gen() -> AsyncIterator[int]:
        yield 1
        raise ValueError("Test error")

    wrapped = immutable_wrap_async_iterator(failing_gen, error_hook=error)

    gen = wrapped()
    assert await gen.__anext__() == 1

    with pytest.raises(ValueError, match="Test error"):
        await gen.__anext__()

    assert captured_error is not None
    assert isinstance(captured_error, ValueError)
    assert str(captured_error) == "Test error"


@pytest.mark.asyncio
async def test_finally_hook_always_called():
    """Test that finally_hook is called regardless of success or failure."""
    finally_call_count = []

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        finally_call_count.append(1)

    async def success_gen() -> AsyncIterator[str]:
        yield "ok"

    async def error_gen() -> AsyncIterator[str]:
        yield "start"
        raise RuntimeError("fail")

    wrapped_success = immutable_wrap_async_iterator(
        success_gen, finally_hook=finally_hook
    )
    [x async for x in wrapped_success()]

    wrapped_error = immutable_wrap_async_iterator(error_gen, finally_hook=finally_hook)
    gen = wrapped_error()
    await gen.__anext__()
    with pytest.raises(RuntimeError):
        await gen.__anext__()

    assert len(finally_call_count) == 2


@pytest.mark.asyncio
async def test_error_hook_receives_context_from_pre_hook():
    """Test that error_hook can access context set by pre_hook."""

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["request_id"] = "12345"

    captured_ctx = {}

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        captured_ctx.update(ctx)

    async def failing_gen() -> AsyncIterator[int]:
        yield 1
        raise Exception("error")

    wrapped = immutable_wrap_async_iterator(failing_gen, pre_hook=pre, error_hook=error)

    gen = wrapped()
    await gen.__anext__()
    with pytest.raises(Exception):
        await gen.__anext__()

    assert captured_ctx["request_id"] == "12345"


@pytest.mark.asyncio
async def test_finally_hook_receives_context():
    """Test that finally_hook receives context from pre_hook."""

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["setup"] = True

    captured_ctx = {}

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        captured_ctx.update(ctx)

    async def dummy() -> AsyncIterator[None]:
        yield

    wrapped = immutable_wrap_async_iterator(
        dummy, pre_hook=pre, finally_hook=finally_hook
    )
    [x async for x in wrapped()]

    assert captured_ctx["setup"] is True


@pytest.mark.asyncio
async def test_post_hook_not_called_on_error():
    """Test that post_hook is not called when async iterator raises an exception."""
    post_called = []

    def post(ctx: Mapping[str, Any]) -> None:
        post_called.append(True)

    async def failing_gen() -> AsyncIterator[int]:
        yield 1
        raise ValueError("error")

    wrapped = immutable_wrap_async_iterator(failing_gen, post_hook=post)

    gen = wrapped()
    await gen.__anext__()
    with pytest.raises(ValueError):
        await gen.__anext__()

    assert len(post_called) == 0


@pytest.mark.asyncio
async def test_complete_lifecycle_success():
    """Test all hooks are called in correct order on success."""
    lifecycle = []

    def pre(ctx: Dict[str, Any]) -> None:
        lifecycle.append("pre")
        ctx["value"] = 1

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        lifecycle.append(f"yield-{value}")

    def post(ctx: Mapping[str, Any]) -> None:
        lifecycle.append("post")

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        lifecycle.append("error")

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        lifecycle.append("finally")

    async def success_gen() -> AsyncIterator[int]:
        lifecycle.append("gen-start")
        yield 1
        yield 2
        lifecycle.append("gen-end")

    wrapped = immutable_wrap_async_iterator(
        success_gen,
        pre_hook=pre,
        yield_hook=yield_hook,
        post_hook=post,
        error_hook=error,
        finally_hook=finally_hook,
    )
    result = [x async for x in wrapped()]

    assert result == [1, 2]
    assert lifecycle == [
        "pre",
        "gen-start",
        "yield-1",
        "yield-2",
        "gen-end",
        "post",
        "finally",
    ]


@pytest.mark.asyncio
async def test_complete_lifecycle_error():
    """Test all hooks are called in correct order on error."""
    lifecycle = []

    def pre(ctx: Dict[str, Any]) -> None:
        lifecycle.append("pre")

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        lifecycle.append(f"yield-{value}")

    def post(ctx: Mapping[str, Any]) -> None:
        lifecycle.append("post")

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        lifecycle.append("error")

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        lifecycle.append("finally")

    async def error_gen() -> AsyncIterator[int]:
        lifecycle.append("gen-start")
        yield 1
        lifecycle.append("gen-before-error")
        raise ValueError("fail")

    wrapped = immutable_wrap_async_iterator(
        error_gen,
        pre_hook=pre,
        yield_hook=yield_hook,
        post_hook=post,
        error_hook=error,
        finally_hook=finally_hook,
    )

    gen = wrapped()
    await gen.__anext__()

    with pytest.raises(ValueError):
        await gen.__anext__()

    assert lifecycle == [
        "pre",
        "gen-start",
        "yield-1",
        "gen-before-error",
        "error",
        "finally",
    ]


@pytest.mark.asyncio
async def test_error_hook_exception_is_caught():
    """Test that exceptions in error_hook don't break error handling."""

    def bad_error_hook(ctx: Mapping[str, Any], err: Exception) -> None:
        raise RuntimeError("Error hook failed")

    async def failing_gen() -> AsyncIterator[int]:
        yield 1
        raise ValueError("Original error")

    wrapped = immutable_wrap_async_iterator(failing_gen, error_hook=bad_error_hook)

    gen = wrapped()
    await gen.__anext__()

    with pytest.raises(ValueError, match="Original error"):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_finally_hook_exception_is_caught():
    """Test that exceptions in finally_hook are caught."""

    def bad_finally_hook(ctx: Mapping[str, Any]) -> None:
        raise RuntimeError("Finally hook failed")

    async def success_gen() -> AsyncIterator[str]:
        yield "ok"

    wrapped = immutable_wrap_async_iterator(success_gen, finally_hook=bad_finally_hook)

    result = [x async for x in wrapped()]
    assert result == ["ok"]


@pytest.mark.asyncio
async def test_yielded_value_not_mutated():
    """Test that yielded values are passed through unchanged."""

    def yield_hook(ctx: Mapping[str, Any], value: Dict[str, int]) -> None:
        value["modified"] = 999

    async def gen_dicts() -> AsyncIterator[Dict[str, int]]:
        yield {"original": 1}
        yield {"original": 2}

    wrapped = immutable_wrap_async_iterator(gen_dicts, yield_hook=yield_hook)
    results = [x async for x in wrapped()]

    assert results[0]["original"] == 1
    assert results[0]["modified"] == 999
    assert results[1]["original"] == 2
    assert results[1]["modified"] == 999


@pytest.mark.asyncio
async def test_iterator_with_complex_types():
    """Test async iterator with complex type annotations."""

    async def complex_gen(data: list[int]) -> AsyncIterator[tuple[int, int]]:
        for i, val in enumerate(data):
            yield (i, val)

    wrapped = immutable_wrap_async_iterator(complex_gen)
    result = [x async for x in wrapped([10, 20, 30])]

    assert result == [(0, 10), (1, 20), (2, 30)]


@pytest.mark.asyncio
async def test_async_execution_order():
    """Test that hooks execute in correct order around async operations."""
    execution_order = []

    def pre(ctx: Dict[str, Any]) -> None:
        execution_order.append("pre")

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        execution_order.append(f"yield-{value}")

    def post(ctx: Mapping[str, Any]) -> None:
        execution_order.append("post")

    async def main() -> AsyncIterator[int]:
        execution_order.append("main-start")
        yield 1
        execution_order.append("main-middle")
        yield 2
        execution_order.append("main-end")

    wrapped = immutable_wrap_async_iterator(
        main, pre_hook=pre, yield_hook=yield_hook, post_hook=post
    )
    result = [x async for x in wrapped()]

    assert result == [1, 2]
    assert execution_order == [
        "pre",
        "main-start",
        "yield-1",
        "main-middle",
        "yield-2",
        "main-end",
        "post",
    ]
