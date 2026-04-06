import logging

from judgeval.utils.decorators.debug_time import debug_time


def test_returns_correct_result():
    @debug_time
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_preserves_function_name():
    @debug_time
    def my_func():
        pass

    assert my_func.__name__ == "my_func"


def test_passes_args_and_kwargs():
    @debug_time
    def func(a, b, c=0):
        return a + b + c

    assert func(1, 2, c=3) == 6


def test_logs_at_debug_level(caplog):
    @debug_time
    def work():
        return 42

    with caplog.at_level(logging.DEBUG, logger="judgeval"):
        work()

    assert any("work" in record.message for record in caplog.records)
