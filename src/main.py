import os

from opentelemetry.sdk.trace.export import BatchSpanProcessor
from judgeval.tracer import Tracer
from judgeval.tracer.exporters import S3Exporter


tracer = Tracer(
    project_name="errors",
)


@tracer.observe
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


@tracer.observe
def error():
    raise Exception("error")


import textwrap


@tracer.observe
def xss():
    return textwrap.dedent(
        """
        # markdown title
        ### hi

        ```py
        def foo():
            print("foo")
        ```
        """
    )


@tracer.observe
def main():
    fib(2)
    xss()


if __name__ == "__main__":
    main()
