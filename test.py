import sys

sys.path = ["./src"] + sys.path

from typing import Any, Generator
from judgeval.common.tracer import TraceClient  # type: ignore
import logging


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


judgment = TraceClient()


def fibonacci(n: int) -> int:
    """
    A simple Fibonacci function to demonstrate tracing.
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def length(s: list[Any]) -> int:
    """
    A simple function to return the length of a list.
    """
    return len(s)


def assert_condition(condition: bool, message: str) -> None:
    """
    A simple assertion function to demonstrate tracing.
    """
    if not condition:
        raise AssertionError(message)


class Foo:
    def test_generator(self, limit: int) -> Generator[int, None, None]:
        """
        A simple generator function to demonstrate tracing.
        """
        while limit > 0:
            yield limit
            limit -= 1


@judgment.observe
def main():
    foo = Foo()
    gen = foo.test_generator(5)
    for value in gen:
        print(value)
    print(length([fibonacci(i) for i in range(6)]))

    assert_condition(length([1, 2, 3]) == 3, "Length should be 3")
    assert_condition(False, "This will raise an assertion error")


with judgment.daemon(deep_tracing=True):
    main()
