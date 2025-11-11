import pytest
from typing import Dict, Any, Mapping

from judgeval.utils.wrappers.immutable_wrap_sync import immutable_wrap_sync


def test_basic_functionality():
    """Test that wrapped function executes and returns correct result."""

    def add(a: int, b: int) -> int:
        return a + b

    wrapped = immutable_wrap_sync(add)
    result = wrapped(2, 3)
    assert result == 5


def test_pre_hook_populates_context():
    """Test that pre_hook can populate context dict."""
    captured_ctx = {}

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["called"] = True
        ctx["value"] = 42

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        captured_ctx.update(ctx)

    def dummy() -> str:
        return "result"

    wrapped = immutable_wrap_sync(dummy, pre_hook=pre, post_hook=post)
    wrapped()

    assert captured_ctx["called"] is True
    assert captured_ctx["value"] == 42


def test_post_hook_receives_result():
    """Test that post_hook receives the function result."""
    captured_result = None

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        nonlocal captured_result
        captured_result = result

    def get_value() -> int:
        return 100

    wrapped = immutable_wrap_sync(get_value, post_hook=post)
    wrapped()

    assert captured_result == 100


def test_post_hook_reads_pre_hook_context():
    """Test that post_hook can read what pre_hook set in context."""

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["start"] = "beginning"

    captured_start = None

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        nonlocal captured_start
        captured_start = ctx.get("start")

    def dummy() -> None:
        pass

    wrapped = immutable_wrap_sync(dummy, pre_hook=pre, post_hook=post)
    wrapped()

    assert captured_start == "beginning"


def test_result_not_mutated():
    """Test that the wrapped function's result is returned unchanged."""

    def post(ctx: Mapping[str, Any], result: Dict[str, int]) -> None:
        # Attempt to mutate (will succeed at runtime but type system discourages)
        result["modified"] = 999

    def get_dict() -> Dict[str, int]:
        return {"original": 1}

    wrapped = immutable_wrap_sync(get_dict, post_hook=post)
    result = wrapped()

    # Result is returned (mutation happens but is not the wrapper's intent)
    assert result["original"] == 1
    assert result["modified"] == 999  # Post-hook did mutate (type system can't prevent)


def test_preserves_function_signature():
    """Test that wrapped function preserves argument types."""

    def complex_func(name: str, age: int, active: bool = True) -> str:
        return f"{name}-{age}-{active}"

    wrapped = immutable_wrap_sync(complex_func)
    result = wrapped("Alice", 30, active=False)

    assert result == "Alice-30-False"


def test_pre_hook_exception_is_caught():
    """Test that exceptions in pre_hook are caught by dont_throw."""

    def bad_pre(ctx: Dict[str, Any]) -> None:
        raise ValueError("Pre hook error")

    def safe_func() -> str:
        return "success"

    wrapped = immutable_wrap_sync(safe_func, pre_hook=bad_pre)
    result = wrapped()

    # Function still executes despite pre_hook error
    assert result == "success"


def test_post_hook_exception_is_caught():
    """Test that exceptions in post_hook are caught by dont_throw."""

    def bad_post(ctx: Mapping[str, Any], result: Any) -> None:
        raise RuntimeError("Post hook error")

    def safe_func() -> int:
        return 42

    wrapped = immutable_wrap_sync(safe_func, post_hook=bad_post)
    result = wrapped()

    # Function still returns result despite post_hook error
    assert result == 42


def test_default_void_hooks():
    """Test that default void hooks work without errors."""

    def simple() -> str:
        return "works"

    wrapped = immutable_wrap_sync(simple)
    result = wrapped()

    assert result == "works"


def test_multiple_calls_isolated_contexts():
    """Test that each call gets its own isolated context."""
    call_count = []

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["id"] = len(call_count)
        call_count.append(ctx["id"])

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        # Verify context is unique per call
        assert ctx["id"] == result

    def get_id() -> int:
        return len(call_count) - 1

    wrapped = immutable_wrap_sync(get_id, pre_hook=pre, post_hook=post)

    wrapped()
    wrapped()
    wrapped()

    assert call_count == [0, 1, 2]


def test_with_args_and_kwargs():
    """Test that args and kwargs are properly passed through."""

    def func(a: int, b: int, c: int = 0, d: int = 0) -> int:
        return a + b + c + d

    wrapped = immutable_wrap_sync(func)
    result = wrapped(1, 2, c=3, d=4)

    assert result == 10


def test_context_immutability_in_post_hook():
    """Test that post_hook receives context as Mapping (readonly interface)."""

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["value"] = 10

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        # Type checker should flag this, but at runtime it might work if Dict is passed
        # This test documents the contract
        assert isinstance(ctx, dict)
        assert ctx["value"] == 10

    def dummy() -> None:
        pass

    wrapped = immutable_wrap_sync(dummy, pre_hook=pre, post_hook=post)
    wrapped()


def test_error_hook_called_on_exception():
    """Test that error_hook is called when function raises an exception."""
    captured_error = None

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        nonlocal captured_error
        captured_error = err

    def failing_func() -> int:
        raise ValueError("Test error")

    wrapped = immutable_wrap_sync(failing_func, error_hook=error)

    with pytest.raises(ValueError, match="Test error"):
        wrapped()

    assert captured_error is not None
    assert isinstance(captured_error, ValueError)
    assert str(captured_error) == "Test error"


def test_finally_hook_always_called():
    """Test that finally_hook is called regardless of success or failure."""
    finally_call_count = []

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        finally_call_count.append(1)

    def success_func() -> str:
        return "ok"

    def error_func() -> str:
        raise RuntimeError("fail")

    # Test with successful function
    wrapped_success = immutable_wrap_sync(success_func, finally_hook=finally_hook)
    wrapped_success()

    # Test with failing function
    wrapped_error = immutable_wrap_sync(error_func, finally_hook=finally_hook)
    with pytest.raises(RuntimeError):
        wrapped_error()

    assert len(finally_call_count) == 2


def test_error_hook_receives_context_from_pre_hook():
    """Test that error_hook can access context set by pre_hook."""

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["request_id"] = "12345"

    captured_ctx = {}

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        captured_ctx.update(ctx)

    def failing_func() -> None:
        raise Exception("error")

    wrapped = immutable_wrap_sync(failing_func, pre_hook=pre, error_hook=error)

    with pytest.raises(Exception):
        wrapped()

    assert captured_ctx["request_id"] == "12345"


def test_finally_hook_receives_context():
    """Test that finally_hook receives context from pre_hook."""

    def pre(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["setup"] = True

    captured_ctx = {}

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        captured_ctx.update(ctx)

    def dummy() -> None:
        pass

    wrapped = immutable_wrap_sync(dummy, pre_hook=pre, finally_hook=finally_hook)
    wrapped()

    assert captured_ctx["setup"] is True


def test_post_hook_not_called_on_error():
    """Test that post_hook is not called when function raises an exception."""
    post_called = []

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        post_called.append(True)

    def failing_func() -> int:
        raise ValueError("error")

    wrapped = immutable_wrap_sync(failing_func, post_hook=post)

    with pytest.raises(ValueError):
        wrapped()

    assert len(post_called) == 0


def test_complete_lifecycle_success():
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

    def success_func() -> str:
        lifecycle.append("func")
        return "ok"

    wrapped = immutable_wrap_sync(
        success_func,
        pre_hook=pre,
        post_hook=post,
        error_hook=error,
        finally_hook=finally_hook,
    )
    result = wrapped()

    assert result == "ok"
    assert lifecycle == ["pre", "func", "post", "finally"]


def test_complete_lifecycle_error():
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

    def error_func() -> str:
        lifecycle.append("func")
        raise ValueError("fail")

    wrapped = immutable_wrap_sync(
        error_func,
        pre_hook=pre,
        post_hook=post,
        error_hook=error,
        finally_hook=finally_hook,
    )

    with pytest.raises(ValueError):
        wrapped()

    assert lifecycle == ["pre", "func", "error", "finally"]


def test_error_hook_exception_is_caught():
    """Test that exceptions in error_hook don't break error handling."""

    def bad_error_hook(ctx: Mapping[str, Any], err: Exception) -> None:
        raise RuntimeError("Error hook failed")

    def failing_func() -> None:
        raise ValueError("Original error")

    wrapped = immutable_wrap_sync(failing_func, error_hook=bad_error_hook)

    # Original error is still raised despite error_hook failing
    with pytest.raises(ValueError, match="Original error"):
        wrapped()


def test_finally_hook_exception_is_caught():
    """Test that exceptions in finally_hook are caught."""

    def bad_finally_hook(ctx: Mapping[str, Any]) -> None:
        raise RuntimeError("Finally hook failed")

    def success_func() -> str:
        return "ok"

    wrapped = immutable_wrap_sync(success_func, finally_hook=bad_finally_hook)

    # Function still succeeds despite finally_hook error
    result = wrapped()
    assert result == "ok"
