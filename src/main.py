import os
import time

from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from judgeval.tracer import Tracer
from judgeval.tracer.exporters import InMemorySpanExporter
from judgeval.tracer.exporters.store import SpanStore

store = SpanStore()

tracer = Tracer(
    project_name="errors",
    processors=[SimpleSpanProcessor(InMemorySpanExporter(store=store))],
)


@tracer.observe
def foo(a: int):
    input("Continue foo?")
    return bar(3 * a)


@tracer.observe
def bar(a: int):
    input("Continue bar?")
    return a + 1


@tracer.observe
def main():
    foo(10)


if __name__ == "__main__":
    main()
