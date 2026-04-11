import httpx
from judgeval import Tracer
from judgeval.trace.propagation import inject

Tracer.init(
    project_name="basic-distributed-tracing",
    resource_attributes={"service.name": "client"},
)


@Tracer.observe(span_type="agent")
def call_server(message: str) -> dict:
    headers: dict = {}
    inject(headers)
    return httpx.post(
        "http://127.0.0.1:8000/run",
        headers=headers,
        json={"message": message},
    ).json()


if __name__ == "__main__":
    print(call_server("hello"))
