from judgeval.utils.wrappers.utils import identity_on_throw


def test_returns_result_on_success():
    def add(ctx, a, b):
        return a + b

    wrapped = identity_on_throw(add)
    assert wrapped({}, 2, 3) == 5


def test_returns_last_arg_on_exception():
    original = [1, 2, 3]

    def mutate(ctx, value):
        raise ValueError("fail")

    wrapped = identity_on_throw(mutate)
    assert wrapped({}, original) is original


def test_returns_last_positional_arg_with_multiple_args():
    sentinel = object()

    def bad_mutate(ctx, intermediate, value):
        raise RuntimeError("crash")

    wrapped = identity_on_throw(bad_mutate)
    result = wrapped({}, "ignored", sentinel)
    assert result is sentinel


def test_preserves_return_type_on_success():
    def transform(ctx, s: str) -> str:
        return s.upper()

    wrapped = identity_on_throw(transform)
    assert wrapped({}, "hello") == "HELLO"


def test_kwargs_do_not_affect_identity_fallback():
    sentinel = "original"

    def bad(ctx, value, extra=None):
        raise Exception("oops")

    wrapped = identity_on_throw(bad)
    result = wrapped({}, sentinel, extra="x")
    assert result is sentinel
