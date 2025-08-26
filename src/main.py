from judgeval.tracer import Tracer
from judgeval.tracer.exporters import InMemorySpanExporter
from judgeval.tracer.exporters.store import SpanStore
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

span_store = SpanStore()
tracer = Tracer(
    project_name="errors",
    processors=[SimpleSpanProcessor(InMemorySpanExporter(span_store))],
)


@tracer.observe(span_type="function")
def fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:

        return fibonacci(n - 1) + fibonacci(n - 2)


print(fibonacci(10))

print(span_store.get_all())
