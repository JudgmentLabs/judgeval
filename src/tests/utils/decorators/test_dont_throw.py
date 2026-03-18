from judgeval.utils.decorators.dont_throw import dont_throw


def test_direct_decoration_returns_value():
    @dont_throw
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_direct_decoration_swallows_exception():
    @dont_throw
    def boom():
        raise ValueError("err")

    assert boom() is None


def test_factory_form_returns_value():
    @dont_throw(default=-1)
    def divide(a, b):
        return a // b

    assert divide(10, 2) == 5


def test_factory_form_returns_default_on_exception():
    @dont_throw(default=-1)
    def boom():
        raise RuntimeError("crash")

    assert boom() == -1


def test_factory_form_with_none_default():
    @dont_throw(default=None)
    def boom():
        raise Exception("x")

    assert boom() is None


def test_preserves_function_name():
    @dont_throw
    def my_func():
        pass

    assert my_func.__name__ == "my_func"


def test_preserves_function_name_factory():
    @dont_throw(default=0)
    def my_func():
        pass

    assert my_func.__name__ == "my_func"


def test_passes_args_and_kwargs():
    @dont_throw
    def func(a, b, c=0):
        return a + b + c

    assert func(1, 2, c=3) == 6


def test_default_object_returned_on_exception():
    sentinel = object()

    @dont_throw(default=sentinel)
    def boom():
        raise Exception

    assert boom() is sentinel
