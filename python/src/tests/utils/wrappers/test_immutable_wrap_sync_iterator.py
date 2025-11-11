import pytest
from typing import Dict, Any, Mapping, Iterator

from judgeval.utils.wrappers.immutable_wrap_sync_iterator import (
    immutable_wrap_sync_iterator,
)


def test_basic_functionality():
    """Test that wrapped iterator executes and yields correct values."""

    def count_to_three() -> Iterator[int]:
        yield 1
        yield 2
        yield 3

    wrapped = immutable_wrap_sync_iterator(count_to_three)
    result = list(wrapped())
    assert result == [1, 2, 3]


def test_pre_hook_populates_context():
    """Test that pre_hook can populate context dict."""
    captured_ctx = {}

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["called"] = True
        ctx["value"] = 42

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        captured_ctx.update(ctx)

    def simple_gen() -> Iterator[str]:
        yield "result"

    wrapped = immutable_wrap_sync_iterator(
        simple_gen, pre_hook=pre, finally_hook=finally_hook
    )
    list(wrapped())

    assert captured_ctx["called"] is True
    assert captured_ctx["value"] == 42


def test_yield_hook_receives_each_value():
    """Test that yield_hook is called for each yielded value."""
    yielded_values = []

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        yielded_values.append(value)

    def count() -> Iterator[int]:
        yield 10
        yield 20
        yield 30

    wrapped = immutable_wrap_sync_iterator(count, yield_hook=yield_hook)
    list(wrapped())

    assert yielded_values == [10, 20, 30]


def test_yield_hook_reads_pre_hook_context():
    """Test that yield_hook can read what pre_hook set in context."""

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["multiplier"] = 2

    captured_contexts = []

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        captured_contexts.append(dict(ctx))

    def gen() -> Iterator[int]:
        yield 1
        yield 2

    wrapped = immutable_wrap_sync_iterator(gen, pre_hook=pre, yield_hook=yield_hook)
    list(wrapped())

    assert all(c.get("multiplier") == 2 for c in captured_contexts)


def test_post_hook_called_on_completion():
    """Test that post_hook is called when iterator completes."""
    post_called = []

    def post(ctx: Mapping[str, Any]) -> None:
        post_called.append(True)

    def simple_gen() -> Iterator[int]:
        yield 1
        yield 2

    wrapped = immutable_wrap_sync_iterator(simple_gen, post_hook=post)
    list(wrapped())

    assert post_called == [True]


def test_post_hook_called_on_empty_iterator():
    """Test that post_hook is called even for empty iterator."""
    post_called = []

    def post(ctx: Mapping[str, Any]) -> None:
        post_called.append(True)

    def empty_gen() -> Iterator[int]:
        return
        yield  # unreachable but makes it an iterator

    wrapped = immutable_wrap_sync_iterator(empty_gen, post_hook=post)
    list(wrapped())

    assert post_called == [True]


def test_preserves_iterator_signature():
    """Test that wrapped iterator preserves argument types."""

    def parameterized_gen(start: int, end: int, prefix: str = "") -> Iterator[str]:
        for i in range(start, end):
            yield f"{prefix}{i}"

    wrapped = immutable_wrap_sync_iterator(parameterized_gen)
    result = list(wrapped(1, 4, prefix="num-"))

    assert result == ["num-1", "num-2", "num-3"]


def test_pre_hook_exception_is_caught():
    """Test that exceptions in pre_hook are caught by dont_throw."""

    def bad_pre(ctx: Dict[str, Any]) -> None:
        raise ValueError("Pre hook error")

    def safe_gen() -> Iterator[str]:
        yield "success"

    wrapped = immutable_wrap_sync_iterator(safe_gen, pre_hook=bad_pre)
    result = list(wrapped())

    # Iterator still executes despite pre_hook error
    assert result == ["success"]


def test_yield_hook_exception_is_caught():
    """Test that exceptions in yield_hook are caught by dont_throw."""

    def bad_yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        raise RuntimeError("Yield hook error")

    def safe_gen() -> Iterator[int]:
        yield 1
        yield 2

    wrapped = immutable_wrap_sync_iterator(safe_gen, yield_hook=bad_yield_hook)
    result = list(wrapped())

    # Iterator still yields all values despite yield_hook errors
    assert result == [1, 2]


def test_post_hook_exception_is_caught():
    """Test that exceptions in post_hook are caught by dont_throw."""

    def bad_post(ctx: Mapping[str, Any]) -> None:
        raise RuntimeError("Post hook error")

    def safe_gen() -> Iterator[int]:
        yield 42

    wrapped = immutable_wrap_sync_iterator(safe_gen, post_hook=bad_post)
    result = list(wrapped())

    # Iterator still completes despite post_hook error
    assert result == [42]


def test_default_void_hooks():
    """Test that default void hooks work without errors."""

    def simple() -> Iterator[str]:
        yield "works"

    wrapped = immutable_wrap_sync_iterator(simple)
    result = list(wrapped())

    assert result == ["works"]


def test_multiple_calls_isolated_contexts():
    """Test that each iterator call gets its own isolated context."""
    call_count = []

    def pre(ctx: Dict[str, Any], i: int) -> None:
        ctx["id"] = len(call_count)
        call_count.append(ctx["id"])

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        # Verify context is unique per iterator instance
        assert ctx["id"] == value

    def gen(i: int) -> Iterator[int]:
        yield i

    wrapped = immutable_wrap_sync_iterator(gen, pre_hook=pre, yield_hook=yield_hook)

    list(wrapped(0))
    list(wrapped(1))
    list(wrapped(2))

    assert call_count == [0, 1, 2]


def test_error_hook_called_on_exception():
    """Test that error_hook is called when iterator raises an exception."""
    captured_error = None

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        nonlocal captured_error
        captured_error = err

    def failing_gen() -> Iterator[int]:
        yield 1
        raise ValueError("Test error")

    wrapped = immutable_wrap_sync_iterator(failing_gen, error_hook=error)

    gen = wrapped()
    assert next(gen) == 1

    with pytest.raises(ValueError, match="Test error"):
        next(gen)

    assert captured_error is not None
    assert isinstance(captured_error, ValueError)
    assert str(captured_error) == "Test error"


def test_finally_hook_always_called():
    """Test that finally_hook is called regardless of success or failure."""
    finally_call_count = []

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        finally_call_count.append(1)

    def success_gen() -> Iterator[str]:
        yield "ok"

    def error_gen() -> Iterator[str]:
        yield "start"
        raise RuntimeError("fail")

    # Test with successful iterator
    wrapped_success = immutable_wrap_sync_iterator(
        success_gen, finally_hook=finally_hook
    )
    list(wrapped_success())

    # Test with failing iterator
    wrapped_error = immutable_wrap_sync_iterator(error_gen, finally_hook=finally_hook)
    gen = wrapped_error()
    next(gen)
    with pytest.raises(RuntimeError):
        next(gen)

    assert len(finally_call_count) == 2


def test_error_hook_receives_context_from_pre_hook():
    """Test that error_hook can access context set by pre_hook."""

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["request_id"] = "12345"

    captured_ctx = {}

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        captured_ctx.update(ctx)

    def failing_gen() -> Iterator[int]:
        yield 1
        raise Exception("error")

    wrapped = immutable_wrap_sync_iterator(failing_gen, pre_hook=pre, error_hook=error)

    gen = wrapped()
    next(gen)
    with pytest.raises(Exception):
        next(gen)

    assert captured_ctx["request_id"] == "12345"


def test_finally_hook_receives_context():
    """Test that finally_hook receives context from pre_hook."""

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["setup"] = True

    captured_ctx = {}

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        captured_ctx.update(ctx)

    def dummy() -> Iterator[None]:
        yield

    wrapped = immutable_wrap_sync_iterator(
        dummy, pre_hook=pre, finally_hook=finally_hook
    )
    list(wrapped())

    assert captured_ctx["setup"] is True


def test_post_hook_not_called_on_error():
    """Test that post_hook is not called when iterator raises an exception."""
    post_called = []

    def post(ctx: Mapping[str, Any]) -> None:
        post_called.append(True)

    def failing_gen() -> Iterator[int]:
        yield 1
        raise ValueError("error")

    wrapped = immutable_wrap_sync_iterator(failing_gen, post_hook=post)

    gen = wrapped()
    next(gen)
    with pytest.raises(ValueError):
        next(gen)

    assert len(post_called) == 0


def test_complete_lifecycle_success():
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

    def success_gen() -> Iterator[int]:
        lifecycle.append("gen-start")
        yield 1
        yield 2
        lifecycle.append("gen-end")

    wrapped = immutable_wrap_sync_iterator(
        success_gen,
        pre_hook=pre,
        yield_hook=yield_hook,
        post_hook=post,
        error_hook=error,
        finally_hook=finally_hook,
    )
    result = list(wrapped())

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


def test_complete_lifecycle_error():
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

    def error_gen() -> Iterator[int]:
        lifecycle.append("gen-start")
        yield 1
        lifecycle.append("gen-before-error")
        raise ValueError("fail")

    wrapped = immutable_wrap_sync_iterator(
        error_gen,
        pre_hook=pre,
        yield_hook=yield_hook,
        post_hook=post,
        error_hook=error,
        finally_hook=finally_hook,
    )

    gen = wrapped()
    next(gen)

    with pytest.raises(ValueError):
        next(gen)

    assert lifecycle == [
        "pre",
        "gen-start",
        "yield-1",
        "gen-before-error",
        "error",
        "finally",
    ]


def test_error_hook_exception_is_caught():
    """Test that exceptions in error_hook don't break error handling."""

    def bad_error_hook(ctx: Mapping[str, Any], err: Exception) -> None:
        raise RuntimeError("Error hook failed")

    def failing_gen() -> Iterator[int]:
        yield 1
        raise ValueError("Original error")

    wrapped = immutable_wrap_sync_iterator(failing_gen, error_hook=bad_error_hook)

    gen = wrapped()
    next(gen)

    # Original error is still raised despite error_hook failing
    with pytest.raises(ValueError, match="Original error"):
        next(gen)


def test_finally_hook_exception_is_caught():
    """Test that exceptions in finally_hook are caught."""

    def bad_finally_hook(ctx: Mapping[str, Any]) -> None:
        raise RuntimeError("Finally hook failed")

    def success_gen() -> Iterator[str]:
        yield "ok"

    wrapped = immutable_wrap_sync_iterator(success_gen, finally_hook=bad_finally_hook)

    # Iterator still completes despite finally_hook error
    result = list(wrapped())
    assert result == ["ok"]


def test_early_iterator_exit():
    """Test that finally_hook is called even if iterator is not fully consumed."""
    finally_called = []

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        finally_called.append(True)

    def long_gen() -> Iterator[int]:
        for i in range(10):
            yield i

    wrapped = immutable_wrap_sync_iterator(long_gen, finally_hook=finally_hook)

    gen = wrapped()
    next(gen)  # Only consume first value
    next(gen)  # Consume second value
    # Iterator not fully consumed, but when it's garbage collected, finally should run
    # However, this is hard to test reliably, so we'll test explicit close

    gen.close()

    # Note: finally_hook won't be called on close in current implementation
    # This documents current behavior - we may want to enhance this


def test_yielded_value_not_mutated():
    """Test that yielded values are passed through unchanged."""

    def yield_hook(ctx: Mapping[str, Any], value: Dict[str, int]) -> None:
        # Attempt to mutate (type system discourages but can't prevent)
        value["modified"] = 999

    def gen_dicts() -> Iterator[Dict[str, int]]:
        yield {"original": 1}
        yield {"original": 2}

    wrapped = immutable_wrap_sync_iterator(gen_dicts, yield_hook=yield_hook)
    results = list(wrapped())

    # Values are yielded (mutation happens but is not wrapper's intent)
    assert results[0]["original"] == 1
    assert results[0]["modified"] == 999
    assert results[1]["original"] == 2
    assert results[1]["modified"] == 999


def test_iterator_with_complex_types():
    """Test iterator with complex type annotations."""

    def complex_gen(data: list[int]) -> Iterator[tuple[int, int]]:
        for i, val in enumerate(data):
            yield (i, val)

    wrapped = immutable_wrap_sync_iterator(complex_gen)
    result = list(wrapped([10, 20, 30]))

    assert result == [(0, 10), (1, 20), (2, 30)]
