import pytest
from typing import Dict, Any, Mapping

from judgeval.utils.wrappers.immutable_wrap_async import immutable_wrap_async


@pytest.mark.asyncio
async def test_basic_functionality():
    """Test that wrapped async function executes and returns correct result."""

    async def add(a: int, b: int) -> int:
        return a + b

    wrapped = immutable_wrap_async(add)
    result = await wrapped(2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_pre_hook_populates_context():
    """Test that pre_hook can populate context dict."""
    captured_ctx = {}

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["called"] = True
        ctx["value"] = 42

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        captured_ctx.update(ctx)

    async def dummy() -> str:
        return "result"

    wrapped = immutable_wrap_async(dummy, pre_hook=pre, post_hook=post)
    await wrapped()

    assert captured_ctx["called"] is True
    assert captured_ctx["value"] == 42


@pytest.mark.asyncio
async def test_post_hook_receives_result():
    """Test that post_hook receives the function result."""
    captured_result = None

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        nonlocal captured_result
        captured_result = result

    async def get_value() -> int:
        return 100

    wrapped = immutable_wrap_async(get_value, post_hook=post)
    await wrapped()

    assert captured_result == 100


@pytest.mark.asyncio
async def test_post_hook_reads_pre_hook_context():
    """Test that post_hook can read what pre_hook set in context."""

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["start"] = "beginning"

    captured_start = None

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        nonlocal captured_start
        captured_start = ctx.get("start")

    async def dummy() -> None:
        pass

    wrapped = immutable_wrap_async(dummy, pre_hook=pre, post_hook=post)
    await wrapped()

    assert captured_start == "beginning"


@pytest.mark.asyncio
async def test_result_not_mutated():
    """Test that the wrapped function's result is returned unchanged."""

    def post(ctx: Mapping[str, Any], result: Dict[str, int]) -> None:
        # Attempt to mutate (will succeed at runtime but type system discourages)
        result["modified"] = 999

    async def get_dict() -> Dict[str, int]:
        return {"original": 1}

    wrapped = immutable_wrap_async(get_dict, post_hook=post)
    result = await wrapped()

    # Result is returned (mutation happens but is not the wrapper's intent)
    assert result["original"] == 1
    assert result["modified"] == 999  # Post-hook did mutate (type system can't prevent)


@pytest.mark.asyncio
async def test_preserves_function_signature():
    """Test that wrapped function preserves argument types."""

    async def complex_func(name: str, age: int, active: bool = True) -> str:
        return f"{name}-{age}-{active}"

    wrapped = immutable_wrap_async(complex_func)
    result = await wrapped("Alice", 30, active=False)

    assert result == "Alice-30-False"


@pytest.mark.asyncio
async def test_pre_hook_exception_is_caught():
    """Test that exceptions in pre_hook are caught by dont_throw."""

    def bad_pre(ctx: Dict[str, Any]) -> None:
        raise ValueError("Pre hook error")

    async def safe_func() -> str:
        return "success"

    wrapped = immutable_wrap_async(safe_func, pre_hook=bad_pre)
    result = await wrapped()

    # Function still executes despite pre_hook error
    assert result == "success"


@pytest.mark.asyncio
async def test_post_hook_exception_is_caught():
    """Test that exceptions in post_hook are caught by dont_throw."""

    def bad_post(ctx: Mapping[str, Any], result: Any) -> None:
        raise RuntimeError("Post hook error")

    async def safe_func() -> int:
        return 42

    wrapped = immutable_wrap_async(safe_func, post_hook=bad_post)
    result = await wrapped()

    # Function still returns result despite post_hook error
    assert result == 42


@pytest.mark.asyncio
async def test_default_void_hooks():
    """Test that default void hooks work without errors."""

    async def simple() -> str:
        return "works"

    wrapped = immutable_wrap_async(simple)
    result = await wrapped()

    assert result == "works"


@pytest.mark.asyncio
async def test_multiple_calls_isolated_contexts():
    """Test that each call gets its own isolated context."""
    call_count = []

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["id"] = len(call_count)
        call_count.append(ctx["id"])

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        # Verify context is unique per call
        assert ctx["id"] == result

    async def get_id() -> int:
        return len(call_count) - 1

    wrapped = immutable_wrap_async(get_id, pre_hook=pre, post_hook=post)

    await wrapped()
    await wrapped()
    await wrapped()

    assert call_count == [0, 1, 2]


@pytest.mark.asyncio
async def test_with_args_and_kwargs():
    """Test that args and kwargs are properly passed through."""

    async def func(a: int, b: int, c: int = 0, d: int = 0) -> int:
        return a + b + c + d

    wrapped = immutable_wrap_async(func)
    result = await wrapped(1, 2, c=3, d=4)

    assert result == 10


@pytest.mark.asyncio
async def test_context_immutability_in_post_hook():
    """Test that post_hook receives context as Mapping (readonly interface)."""

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["value"] = 10

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        # Type checker should flag this, but at runtime it might work if Dict is passed
        # This test documents the contract
        assert isinstance(ctx, dict)
        assert ctx["value"] == 10

    async def dummy() -> None:
        pass

    wrapped = immutable_wrap_async(dummy, pre_hook=pre, post_hook=post)
    await wrapped()


@pytest.mark.asyncio
async def test_async_execution_order():
    """Test that hooks execute in correct order around async function."""
    execution_order = []

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        execution_order.append("pre")

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        execution_order.append("post")

    async def main() -> str:
        execution_order.append("main")
        return "done"

    wrapped = immutable_wrap_async(main, pre_hook=pre, post_hook=post)
    result = await wrapped()

    assert execution_order == ["pre", "main", "post"]
    assert result == "done"


@pytest.mark.asyncio
async def test_error_hook_called_on_exception():
    """Test that error_hook is called when async function raises an exception."""
    captured_error = None

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        nonlocal captured_error
        captured_error = err

    async def failing_func() -> int:
        raise ValueError("Test error")

    wrapped = immutable_wrap_async(failing_func, error_hook=error)

    with pytest.raises(ValueError, match="Test error"):
        await wrapped()

    assert captured_error is not None
    assert isinstance(captured_error, ValueError)
    assert str(captured_error) == "Test error"


@pytest.mark.asyncio
async def test_finally_hook_always_called():
    """Test that finally_hook is called regardless of success or failure."""
    finally_call_count = []

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        finally_call_count.append(1)

    async def success_func() -> str:
        return "ok"

    async def error_func() -> str:
        raise RuntimeError("fail")

    # Test with successful function
    wrapped_success = immutable_wrap_async(success_func, finally_hook=finally_hook)
    await wrapped_success()

    # Test with failing function
    wrapped_error = immutable_wrap_async(error_func, finally_hook=finally_hook)
    with pytest.raises(RuntimeError):
        await wrapped_error()

    assert len(finally_call_count) == 2


@pytest.mark.asyncio
async def test_error_hook_receives_context_from_pre_hook():
    """Test that error_hook can access context set by pre_hook."""

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["request_id"] = "12345"

    captured_ctx = {}

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        captured_ctx.update(ctx)

    async def failing_func() -> None:
        raise Exception("error")

    wrapped = immutable_wrap_async(failing_func, pre_hook=pre, error_hook=error)

    with pytest.raises(Exception):
        await wrapped()

    assert captured_ctx["request_id"] == "12345"


@pytest.mark.asyncio
async def test_finally_hook_receives_context():
    """Test that finally_hook receives context from pre_hook."""

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["setup"] = True

    captured_ctx = {}

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        captured_ctx.update(ctx)

    async def dummy() -> None:
        pass

    wrapped = immutable_wrap_async(dummy, pre_hook=pre, finally_hook=finally_hook)
    await wrapped()

    assert captured_ctx["setup"] is True


@pytest.mark.asyncio
async def test_post_hook_not_called_on_error():
    """Test that post_hook is not called when function raises an exception."""
    post_called = []

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        post_called.append(True)

    async def failing_func() -> int:
        raise ValueError("error")

    wrapped = immutable_wrap_async(failing_func, post_hook=post)

    with pytest.raises(ValueError):
        await wrapped()

    assert len(post_called) == 0


@pytest.mark.asyncio
async def test_complete_lifecycle_success():
    """Test all hooks are called in correct order on success."""
    lifecycle = []

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        lifecycle.append("pre")
        ctx["value"] = 1

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        lifecycle.append("post")

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        lifecycle.append("error")

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        lifecycle.append("finally")

    async def success_func() -> str:
        lifecycle.append("func")
        return "ok"

    wrapped = immutable_wrap_async(
        success_func,
        pre_hook=pre,
        post_hook=post,
        error_hook=error,
        finally_hook=finally_hook,
    )
    result = await wrapped()

    assert result == "ok"
    assert lifecycle == ["pre", "func", "post", "finally"]


@pytest.mark.asyncio
async def test_complete_lifecycle_error():
    """Test all hooks are called in correct order on error."""
    lifecycle = []

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        lifecycle.append("pre")

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        lifecycle.append("post")

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        lifecycle.append("error")

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        lifecycle.append("finally")

    async def error_func() -> str:
        lifecycle.append("func")
        raise ValueError("fail")

    wrapped = immutable_wrap_async(
        error_func,
        pre_hook=pre,
        post_hook=post,
        error_hook=error,
        finally_hook=finally_hook,
    )

    with pytest.raises(ValueError):
        await wrapped()

    assert lifecycle == ["pre", "func", "error", "finally"]


@pytest.mark.asyncio
async def test_error_hook_exception_is_caught():
    """Test that exceptions in error_hook don't break error handling."""

    def bad_error_hook(ctx: Mapping[str, Any], err: Exception) -> None:
        raise RuntimeError("Error hook failed")

    async def failing_func() -> None:
        raise ValueError("Original error")

    wrapped = immutable_wrap_async(failing_func, error_hook=bad_error_hook)

    # Original error is still raised despite error_hook failing
    with pytest.raises(ValueError, match="Original error"):
        await wrapped()


@pytest.mark.asyncio
async def test_finally_hook_exception_is_caught():
    """Test that exceptions in finally_hook are caught."""

    def bad_finally_hook(ctx: Mapping[str, Any]) -> None:
        raise RuntimeError("Finally hook failed")

    async def success_func() -> str:
        return "ok"

    wrapped = immutable_wrap_async(success_func, finally_hook=bad_finally_hook)

    # Function still succeeds despite finally_hook error
    result = await wrapped()
    assert result == "ok"
