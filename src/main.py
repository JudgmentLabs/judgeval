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


# Test order

import time


@tracer.observe
def call1():
    time.sleep(1)
    return 1


@tracer.observe
def call2():
    time.sleep(1)
    return 2


@tracer.observe
def call3():
    time.sleep(1)
    return 3


@tracer.observe
def call4():
    time.sleep(1)
    return 4


@tracer.observe
def call5():
    time.sleep(1)
    return 5


@tracer.observe
def main():
    fib(2)

    call1()
    call2()
    call3()
    call4()
    call5()

    error()


if __name__ == "__main__":
    main()
