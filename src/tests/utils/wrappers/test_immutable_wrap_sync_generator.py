import pytest
from typing import Dict, Any, Mapping, Generator

from judgeval.utils.wrappers.immutable_wrap_sync_generator import (
    immutable_wrap_sync_generator,
)


def test_basic_functionality():
    """Test that wrapped generator executes and yields correct values."""

    def count_to_three() -> Generator[int, None, None]:
        yield 1
        yield 2
        yield 3

    wrapped = immutable_wrap_sync_generator(count_to_three)
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

    def simple_gen() -> Generator[str, None, None]:
        yield "result"

    wrapped = immutable_wrap_sync_generator(
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

    def count() -> Generator[int, None, None]:
        yield 10
        yield 20
        yield 30

    wrapped = immutable_wrap_sync_generator(count, yield_hook=yield_hook)
    list(wrapped())

    assert yielded_values == [10, 20, 30]


def test_yield_hook_reads_pre_hook_context():
    """Test that yield_hook can read what pre_hook set in context."""

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["multiplier"] = 2

    captured_contexts = []

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        captured_contexts.append(dict(ctx))

    def gen() -> Generator[int, None, None]:
        yield 1
        yield 2

    wrapped = immutable_wrap_sync_generator(gen, pre_hook=pre, yield_hook=yield_hook)
    list(wrapped())

    assert all(c.get("multiplier") == 2 for c in captured_contexts)


def test_post_hook_receives_return_value():
    """Test that post_hook receives the generator return value."""
    captured_return = None

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        nonlocal captured_return
        captured_return = result

    def gen_with_return() -> Generator[int, None, str]:
        yield 1
        yield 2
        return "done"

    wrapped = immutable_wrap_sync_generator(gen_with_return, post_hook=post)
    list(wrapped())

    assert captured_return == "done"


def test_post_hook_called_on_empty_generator():
    """Test that post_hook is called even for empty generator."""
    post_called = []

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        post_called.append(result)

    def empty_gen() -> Generator[int, None, str]:
        return "empty"
        yield  # unreachable but makes it a generator

    wrapped = immutable_wrap_sync_generator(empty_gen, post_hook=post)
    list(wrapped())

    assert post_called == ["empty"]


def test_preserves_generator_signature():
    """Test that wrapped generator preserves argument types."""

    def parameterized_gen(
        start: int, end: int, prefix: str = ""
    ) -> Generator[str, None, None]:
        for i in range(start, end):
            yield f"{prefix}{i}"

    wrapped = immutable_wrap_sync_generator(parameterized_gen)
    result = list(wrapped(1, 4, prefix="num-"))

    assert result == ["num-1", "num-2", "num-3"]


def test_pre_hook_exception_is_caught():
    """Test that exceptions in pre_hook are caught by dont_throw."""

    def bad_pre(ctx: Dict[str, Any]) -> None:
        raise ValueError("Pre hook error")

    def safe_gen() -> Generator[str, None, None]:
        yield "success"

    wrapped = immutable_wrap_sync_generator(safe_gen, pre_hook=bad_pre)
    result = list(wrapped())

    # Generator still executes despite pre_hook error
    assert result == ["success"]


def test_yield_hook_exception_is_caught():
    """Test that exceptions in yield_hook are caught by dont_throw."""

    def bad_yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        raise RuntimeError("Yield hook error")

    def safe_gen() -> Generator[int, None, None]:
        yield 1
        yield 2

    wrapped = immutable_wrap_sync_generator(safe_gen, yield_hook=bad_yield_hook)
    result = list(wrapped())

    # Generator still yields all values despite yield_hook errors
    assert result == [1, 2]


def test_post_hook_exception_is_caught():
    """Test that exceptions in post_hook are caught by dont_throw."""

    def bad_post(ctx: Mapping[str, Any], result: Any) -> None:
        raise RuntimeError("Post hook error")

    def safe_gen() -> Generator[int, None, int]:
        yield 42
        return 100

    wrapped = immutable_wrap_sync_generator(safe_gen, post_hook=bad_post)
    result = list(wrapped())

    # Generator still completes despite post_hook error
    assert result == [42]


def test_default_void_hooks():
    """Test that default void hooks work without errors."""

    def simple() -> Generator[str, None, None]:
        yield "works"

    wrapped = immutable_wrap_sync_generator(simple)
    result = list(wrapped())

    assert result == ["works"]


def test_multiple_calls_isolated_contexts():
    """Test that each generator call gets its own isolated context."""
    call_count = []

    def pre(ctx: Dict[str, Any]) -> None:
        ctx["id"] = len(call_count)
        call_count.append(ctx["id"])

    def yield_hook(ctx: Mapping[str, Any], value: Any) -> None:
        # Verify context is unique per generator instance
        assert ctx["id"] == value

    def gen(i: int) -> Generator[int, None, None]:
        yield i

    wrapped = immutable_wrap_sync_generator(gen, pre_hook=pre, yield_hook=yield_hook)

    list(wrapped(0))
    list(wrapped(1))
    list(wrapped(2))

    assert call_count == [0, 1, 2]


def test_error_hook_called_on_exception():
    """Test that error_hook is called when generator raises an exception."""
    captured_error = None

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        nonlocal captured_error
        captured_error = err

    def failing_gen() -> Generator[int, None, None]:
        yield 1
        raise ValueError("Test error")

    wrapped = immutable_wrap_sync_generator(failing_gen, error_hook=error)

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

    def success_gen() -> Generator[str, None, None]:
        yield "ok"

    def error_gen() -> Generator[str, None, None]:
        yield "start"
        raise RuntimeError("fail")

    # Test with successful generator
    wrapped_success = immutable_wrap_sync_generator(
        success_gen, finally_hook=finally_hook
    )
    list(wrapped_success())

    # Test with failing generator
    wrapped_error = immutable_wrap_sync_generator(error_gen, finally_hook=finally_hook)
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

    def failing_gen() -> Generator[int, None, None]:
        yield 1
        raise Exception("error")

    wrapped = immutable_wrap_sync_generator(failing_gen, pre_hook=pre, error_hook=error)

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

    def dummy() -> Generator[None, None, None]:
        yield

    wrapped = immutable_wrap_sync_generator(
        dummy, pre_hook=pre, finally_hook=finally_hook
    )
    list(wrapped())

    assert captured_ctx["setup"] is True


def test_post_hook_not_called_on_error():
    """Test that post_hook is not called when generator raises an exception."""
    post_called = []

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        post_called.append(True)

    def failing_gen() -> Generator[int, None, None]:
        yield 1
        raise ValueError("error")

    wrapped = immutable_wrap_sync_generator(failing_gen, post_hook=post)

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

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        lifecycle.append("post")

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        lifecycle.append("error")

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        lifecycle.append("finally")

    def success_gen() -> Generator[int, None, str]:
        lifecycle.append("gen-start")
        yield 1
        yield 2
        lifecycle.append("gen-end")
        return "ok"

    wrapped = immutable_wrap_sync_generator(
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

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        lifecycle.append("post")

    def error(ctx: Mapping[str, Any], err: Exception) -> None:
        lifecycle.append("error")

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        lifecycle.append("finally")

    def error_gen() -> Generator[int, None, None]:
        lifecycle.append("gen-start")
        yield 1
        lifecycle.append("gen-before-error")
        raise ValueError("fail")

    wrapped = immutable_wrap_sync_generator(
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

    def failing_gen() -> Generator[int, None, None]:
        yield 1
        raise ValueError("Original error")

    wrapped = immutable_wrap_sync_generator(failing_gen, error_hook=bad_error_hook)

    gen = wrapped()
    next(gen)

    # Original error is still raised despite error_hook failing
    with pytest.raises(ValueError, match="Original error"):
        next(gen)


def test_finally_hook_exception_is_caught():
    """Test that exceptions in finally_hook are caught."""

    def bad_finally_hook(ctx: Mapping[str, Any]) -> None:
        raise RuntimeError("Finally hook failed")

    def success_gen() -> Generator[str, None, None]:
        yield "ok"

    wrapped = immutable_wrap_sync_generator(success_gen, finally_hook=bad_finally_hook)

    # Generator still completes despite finally_hook error
    result = list(wrapped())
    assert result == ["ok"]


def test_early_generator_exit():
    """Test that finally_hook is called even if generator is not fully consumed."""
    finally_called = []

    def finally_hook(ctx: Mapping[str, Any]) -> None:
        finally_called.append(True)

    def long_gen() -> Generator[int, None, None]:
        for i in range(10):
            yield i

    wrapped = immutable_wrap_sync_generator(long_gen, finally_hook=finally_hook)

    gen = wrapped()
    next(gen)  # Only consume first value
    next(gen)  # Consume second value
    # Generator not fully consumed, but when it's garbage collected, finally should run
    # However, this is hard to test reliably, so we'll test explicit close

    gen.close()

    # Note: finally_hook won't be called on close in current implementation
    # This documents current behavior - we may want to enhance this


def test_generator_return_none():
    """Test generator that implicitly returns None."""
    post_called = []

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        post_called.append(result)

    def simple_gen() -> Generator[int, None, None]:
        yield 1
        yield 2

    wrapped = immutable_wrap_sync_generator(simple_gen, post_hook=post)
    list(wrapped())

    assert post_called == [None]


def test_yielded_value_not_mutated():
    """Test that yielded values are passed through unchanged."""

    def yield_hook(ctx: Mapping[str, Any], value: Dict[str, int]) -> None:
        # Attempt to mutate (type system discourages but can't prevent)
        value["modified"] = 999

    def gen_dicts() -> Generator[Dict[str, int], None, None]:
        yield {"original": 1}
        yield {"original": 2}

    wrapped = immutable_wrap_sync_generator(gen_dicts, yield_hook=yield_hook)
    results = list(wrapped())

    # Values are yielded (mutation happens but is not wrapper's intent)
    assert results[0]["original"] == 1
    assert results[0]["modified"] == 999
    assert results[1]["original"] == 2
    assert results[1]["modified"] == 999


def test_send_support():
    """Test that generator send() is properly supported."""

    def echo_gen() -> Generator[int, int, None]:
        value = yield 1
        assert value == 10
        value = yield 2
        assert value == 20

    wrapped = immutable_wrap_sync_generator(echo_gen)
    gen = wrapped()

    assert next(gen) == 1
    assert gen.send(10) == 2
    with pytest.raises(StopIteration):
        gen.send(20)


def test_generator_with_complex_types():
    """Test generator with complex type annotations."""

    def complex_gen(data: list[int]) -> Generator[tuple[int, int], None, int]:
        total = 0
        for i, val in enumerate(data):
            yield (i, val)
            total += val
        return total

    captured_return = None

    def post(ctx: Mapping[str, Any], result: Any) -> None:
        nonlocal captured_return
        captured_return = result

    wrapped = immutable_wrap_sync_generator(complex_gen, post_hook=post)
    result = list(wrapped([10, 20, 30]))

    assert result == [(0, 10), (1, 20), (2, 30)]
    assert captured_return == 60
