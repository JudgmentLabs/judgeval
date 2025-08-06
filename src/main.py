from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any

from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from judgeval.tracer import Tracer, wrap
from judgeval.tracer.exporters import InMemorySpanExporter
from judgeval.tracer.exporters.store import ABCSpanStore, SpanStore

import openai


@dataclass
class User:
    id: int
    name: str
    email: str
    active: bool = True


class CustomObject:
    def __init__(self, value: str, count: int):
        self.value = value
        self.count = count
        self.created_at = datetime.now()


spans: ABCSpanStore = SpanStore()

tracer = Tracer(
    api_url="http://localhost:8000/otel/v1/traces",
    project_name="test",
    processors=[
        SimpleSpanProcessor(InMemorySpanExporter(store=spans)),
    ],
    enable_monitoring=True,
)

client = wrap(openai.OpenAI())


@tracer.observe
def return_string() -> str:
    return "hello world"


@tracer.observe
def return_integer() -> int:
    return 42


@tracer.observe
def return_float() -> float:
    return 3.14159


@tracer.observe
def return_boolean() -> bool:
    return True


@tracer.observe
def return_none() -> None:
    return None


@tracer.observe
def return_list() -> List[str]:
    return ["apple", "banana", "cherry"]


@tracer.observe
def return_dict() -> Dict[str, Any]:
    return {"name": "John", "age": 30, "scores": [85, 92, 78], "active": True}


@tracer.observe
def return_nested_data() -> Dict[str, Any]:
    return {
        "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "metadata": {"total": 2, "timestamp": datetime.now().isoformat()},
    }


@tracer.observe
def return_dataclass() -> User:
    return User(id=123, name="Jane Doe", email="jane@example.com", active=True)


@tracer.observe
def return_custom_object() -> CustomObject:
    return CustomObject("test_value", 10)


@tracer.observe
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@tracer.observe
def process_user_data(user: User, multiplier: int = 2) -> Dict[str, Any]:
    return {
        "user_id": user.id * multiplier,
        "display_name": user.name.upper(),
        "is_valid": user.active and "@" in user.email,
    }


@tracer.observe
def chain_functions(value: str) -> str:
    result = return_string()
    return f"{result} - {value}"


@tracer.observe
def raise_exception() -> str:
    raise ValueError("This is a test exception")


@tracer.observe
def main():
    print("Testing various data types with @observe decorator")

    print("\n1. Basic data types:")
    print(f"String: {return_string()}")
    print(f"Integer: {return_integer()}")
    print(f"Float: {return_float()}")
    print(f"Boolean: {return_boolean()}")
    print(f"None: {return_none()}")

    print("\n2. Collections:")
    print(f"List: {return_list()}")
    print(f"Dict: {return_dict()}")
    print(f"Nested: {return_nested_data()}")

    print("\n3. Objects:")
    user = return_dataclass()
    print(f"Dataclass: {user}")

    custom = return_custom_object()
    print(f"Custom object: {custom.value}, {custom.count}")

    print("\n4. Function with arguments:")
    processed = process_user_data(user, 3)
    print(f"Processed: {processed}")

    print("\n5. Recursive function:")
    fib_result = fibonacci(5)
    print(f"Fibonacci(5): {fib_result}")

    print("\n6. Function chaining:")
    chained = chain_functions("additional")
    print(f"Chained: {chained}")

    print("\n7. Exception handling:")
    try:
        raise_exception()
    except ValueError as e:
        print(f"Caught exception: {e}")

    tracer.force_flush()
    print(f"\nTotal spans collected: {len(spans.get_all())}")


if __name__ == "__main__":
    print(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
    )
