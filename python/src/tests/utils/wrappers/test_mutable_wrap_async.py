import pytest
from typing import Dict, Any

from judgeval.utils.wrappers.mutable_wrap_async import mutable_wrap_async


@pytest.mark.asyncio
async def test_basic_functionality():
    """Test that wrapped async function executes and returns correct result."""

    async def add(a: int, b: int) -> int:
        return a + b

    wrapped = mutable_wrap_async(add)
    result = await wrapped(2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_mutate_kwargs_hook_modifies_kwargs():
    """Test that mutate_kwargs_hook can modify kwargs before function execution."""
    call_log = []

    def mutate_kwargs(ctx: Dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        call_log.append(("mutate_kwargs", dict(kwargs)))
        modified = dict(kwargs)
        modified["value"] = kwargs.get("value", 0) + 10
        return modified

    async def func(value: int = 0) -> int:
        call_log.append(("func", value))
        return value

    wrapped = mutable_wrap_async(func, mutate_kwargs_hook=mutate_kwargs)
    result = await wrapped(value=5)

    assert result == 15
    assert call_log == [("mutate_kwargs", {"value": 5}), ("func", 15)]


@pytest.mark.asyncio
async def test_mutate_args_hook_modifies_args():
    """Test that mutate_args_hook can modify args before function execution."""
    call_log = []

    def mutate_args(ctx: Dict[str, Any], args: tuple[Any, ...]) -> tuple[Any, ...]:
        call_log.append(("mutate_args", args))
        return tuple(x + 10 for x in args)

    async def func(a: int, b: int) -> int:
        call_log.append(("func", a, b))
        return a + b

    wrapped = mutable_wrap_async(func, mutate_args_hook=mutate_args)
    result = await wrapped(5, 3)

    assert result == 28  # (5+10) + (3+10)
    assert call_log == [("mutate_args", (5, 3)), ("func", 15, 13)]


@pytest.mark.asyncio
async def test_mutate_kwargs_hook_failure_uses_identity():
    """Test that if mutate_kwargs_hook raises, it falls back to identity (original kwargs)."""
    call_log = []

    def failing_mutate_kwargs(
        ctx: Dict[str, Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        call_log.append("mutate_kwargs_called")
        raise RuntimeError("mutate_kwargs failed!")

    async def func(value: int = 0) -> int:
        call_log.append(("func", value))
        return value

    wrapped = mutable_wrap_async(func, mutate_kwargs_hook=failing_mutate_kwargs)
    result = await wrapped(value=5)

    assert result == 5  # Original kwargs used
    assert "mutate_kwargs_called" in call_log
    assert ("func", 5) in call_log


@pytest.mark.asyncio
async def test_mutate_args_hook_failure_uses_identity():
    """Test that if mutate_args_hook raises, it falls back to identity (original args)."""
    call_log = []

    def failing_mutate_args(
        ctx: Dict[str, Any], args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        call_log.append("mutate_args_called")
        raise RuntimeError("mutate_args failed!")

    async def func(a: int, b: int) -> int:
        call_log.append(("func", a, b))
        return a + b

    wrapped = mutable_wrap_async(func, mutate_args_hook=failing_mutate_args)
    result = await wrapped(5, 3)

    assert result == 8  # Original args used
    assert "mutate_args_called" in call_log
    assert ("func", 5, 3) in call_log


@pytest.mark.asyncio
async def test_mutate_both_args_and_kwargs():
    """Test that both mutate_args_hook and mutate_kwargs_hook work together."""
    call_log = []

    def mutate_args(ctx: Dict[str, Any], args: tuple[Any, ...]) -> tuple[Any, ...]:
        call_log.append(("mutate_args", args))
        return tuple(x * 2 for x in args)

    def mutate_kwargs(ctx: Dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        call_log.append(("mutate_kwargs", dict(kwargs)))
        modified = dict(kwargs)
        modified["multiplier"] = kwargs.get("multiplier", 1) * 3
        return modified

    async def func(a: int, b: int, multiplier: int = 1) -> int:
        call_log.append(("func", a, b, multiplier))
        return (a + b) * multiplier

    wrapped = mutable_wrap_async(
        func, mutate_args_hook=mutate_args, mutate_kwargs_hook=mutate_kwargs
    )
    result = await wrapped(5, 3, multiplier=2)

    # args: (5, 3) -> (10, 6)
    # kwargs: {multiplier: 2} -> {multiplier: 6}
    # func: (10 + 6) * 6 = 96
    assert result == 96
    assert call_log == [
        ("mutate_args", (5, 3)),
        ("mutate_kwargs", {"multiplier": 2}),
        ("func", 10, 6, 6),
    ]


@pytest.mark.asyncio
async def test_mutate_kwargs_hook_with_context():
    """Test that mutate_kwargs_hook can access and use context."""

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["add_value"] = 100

    def mutate_kwargs(ctx: Dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        modified = dict(kwargs)
        modified["value"] = kwargs.get("value", 0) + ctx.get("add_value", 0)
        return modified

    async def func(value: int = 0) -> int:
        return value

    wrapped = mutable_wrap_async(
        func, pre_hook=pre_hook, mutate_kwargs_hook=mutate_kwargs
    )
    result = await wrapped(value=5)

    assert result == 105


@pytest.mark.asyncio
async def test_mutate_hook_with_mutate_kwargs():
    """Test that mutate_hook works alongside mutate_kwargs_hook."""
    call_log = []

    def mutate_kwargs(ctx: Dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        call_log.append("mutate_kwargs")
        modified = dict(kwargs)
        modified["value"] = kwargs.get("value", 0) + 10
        return modified

    def mutate_result(ctx: Dict[str, Any], result: int) -> int:
        call_log.append("mutate_result")
        return result * 2

    async def func(value: int = 0) -> int:
        call_log.append("func")
        return value

    wrapped = mutable_wrap_async(
        func, mutate_kwargs_hook=mutate_kwargs, mutate_hook=mutate_result
    )
    result = await wrapped(value=5)

    # kwargs mutated: 5 -> 15
    # func returns: 15
    # result mutated: 15 -> 30
    assert result == 30
    assert call_log == ["mutate_kwargs", "func", "mutate_result"]


@pytest.mark.asyncio
async def test_pre_hook_called_before_kwargs_mutation():
    """Test that pre_hook is called before kwargs are mutated."""
    call_order = []

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        call_order.append(("pre_hook", kwargs.copy()))
        ctx["original_value"] = kwargs.get("value", 0)

    def mutate_kwargs(ctx: Dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        call_order.append(("mutate_kwargs", kwargs.copy(), ctx.get("original_value")))
        modified = dict(kwargs)
        modified["value"] = kwargs.get("value", 0) + 10
        return modified

    async def func(value: int = 0) -> int:
        call_order.append(("func", value))
        return value

    wrapped = mutable_wrap_async(
        func, pre_hook=pre_hook, mutate_kwargs_hook=mutate_kwargs
    )
    result = await wrapped(value=5)

    assert result == 15
    assert call_order[0] == ("pre_hook", {"value": 5})
    assert call_order[1] == ("mutate_kwargs", {"value": 5}, 5)
    assert call_order[2] == ("func", 15)


@pytest.mark.asyncio
async def test_error_hook_called_when_function_fails():
    """Test that error_hook is called when the wrapped async function raises."""
    error_captured = {}

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        error_captured["error"] = error
        error_captured["type"] = type(error).__name__

    async def failing_func() -> int:
        raise ValueError("Test error")

    wrapped = mutable_wrap_async(failing_func, error_hook=error_hook)

    with pytest.raises(ValueError, match="Test error"):
        await wrapped()

    assert error_captured["type"] == "ValueError"
    assert str(error_captured["error"]) == "Test error"


@pytest.mark.asyncio
async def test_finally_hook_called_on_success():
    """Test that finally_hook is called even on success."""
    finally_called = []

    def finally_hook(ctx: Dict[str, Any]) -> None:
        finally_called.append(True)

    async def func() -> str:
        return "success"

    wrapped = mutable_wrap_async(func, finally_hook=finally_hook)
    result = await wrapped()

    assert result == "success"
    assert finally_called == [True]


@pytest.mark.asyncio
async def test_finally_hook_called_on_failure():
    """Test that finally_hook is called even when function fails."""
    finally_called = []

    def finally_hook(ctx: Dict[str, Any]) -> None:
        finally_called.append(True)

    async def failing_func() -> int:
        raise RuntimeError("fail")

    wrapped = mutable_wrap_async(failing_func, finally_hook=finally_hook)

    with pytest.raises(RuntimeError):
        await wrapped()

    assert finally_called == [True]


@pytest.mark.asyncio
async def test_kwargs_mutation_preserves_other_parameters():
    """Test that mutate_kwargs_hook preserves parameters it doesn't modify."""

    def mutate_kwargs(ctx: Dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        modified = dict(kwargs)
        modified["b"] = kwargs.get("b", 0) + 10
        return modified

    async def func(a: int, b: int, c: int) -> int:
        return a + b + c

    wrapped = mutable_wrap_async(func, mutate_kwargs_hook=mutate_kwargs)
    result = await wrapped(a=1, b=2, c=3)

    assert result == 1 + 12 + 3  # Only b was modified


@pytest.mark.asyncio
async def test_empty_kwargs_mutation():
    """Test that mutate_kwargs_hook works with empty kwargs."""
    call_log = []

    def mutate_kwargs(ctx: Dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        call_log.append(("mutate_kwargs", dict(kwargs)))
        return {"default": 42}

    async def func(default: int = 0) -> int:
        call_log.append(("func", default))
        return default

    wrapped = mutable_wrap_async(func, mutate_kwargs_hook=mutate_kwargs)
    result = await wrapped()

    assert result == 42
    assert call_log == [("mutate_kwargs", {}), ("func", 42)]


@pytest.mark.asyncio
async def test_mutate_hook_failure_returns_original_result():
    """Test that if mutate_hook fails, original result is returned."""

    def mutate_result(ctx: Dict[str, Any], result: int) -> int:
        raise RuntimeError("mutate failed")

    async def func() -> int:
        return 42

    wrapped = mutable_wrap_async(func, mutate_hook=mutate_result)
    result = await wrapped()

    # Should return original result when mutate_hook fails
    assert result == 42
