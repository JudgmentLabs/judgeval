import json
import orjson
import timeit
import random
import string
import uuid
from datetime import datetime, timezone

def generate_random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def create_span_data(size=100):
    spans = []
    for _ in range(size):
        spans.append({
            "span_id": str(uuid.uuid4()),
            "trace_id": str(uuid.uuid4()),
            "depth": random.randint(0, 5),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {generate_random_string(5): random.random() for _ in range(3)},
            "output": {"result": generate_random_string(20)},
            "error": None,
            "parent_span_id": str(uuid.uuid4()),
            "function": f"my_module.{generate_random_string(8)}",
            "duration": random.uniform(0.1, 5.0),
            "span_type": random.choice(["span", "tool", "llm", "evaluation", "chain"]),
            "usage": {
                "completion_tokens": random.randint(50, 500),
                "prompt_tokens": random.randint(100, 1000),
                "total_tokens": random.randint(150, 1500),
                "cost": random.uniform(0.0001, 0.005),
                "model_name": random.choice(["gpt-4", "claude-3-opus", "gemini-1.5-pro"])
            },
            "has_evaluation": random.choice([True, False]),
            "agent_name": f"Agent-{generate_random_string(5)}",
            "state_before": {"status": "running", "step": random.randint(1, 10)},
            "state_after": {"status": "completed", "step": random.randint(1, 10) + 1},
            "additional_metadata": {generate_random_string(10): generate_random_string(25) for _ in range(2)},
            "update_id": random.randint(1, 100)
        })
    return spans

data_to_serialize = create_span_data(size=100)
number_of_executions = 10000

json_string = json.dumps(data_to_serialize)
orjson_bytes = orjson.dumps(data_to_serialize)

print(f"Running benchmark with {number_of_executions} executions...")

time_json_dumps = timeit.timeit(lambda: json.dumps(data_to_serialize), number=number_of_executions)
time_orjson_dumps = timeit.timeit(lambda: orjson.dumps(data_to_serialize), number=number_of_executions)
time_json_loads = timeit.timeit(lambda: json.loads(json_string), number=number_of_executions)
time_orjson_loads = timeit.timeit(lambda: orjson.loads(orjson_bytes), number=number_of_executions)

dumps_speedup = time_json_dumps / time_orjson_dumps if time_orjson_dumps > 0 else float('inf')
loads_speedup = time_json_loads / time_orjson_loads if time_orjson_loads > 0 else float('inf')

print("\n--- Serialization (dumps) ---")
print(f"json:   {time_json_dumps:.4f}s")
print(f"orjson: {time_orjson_dumps:.4f}s")
print(f"Speedup: {dumps_speedup:.2f}x")

print("\n--- Deserialization (loads) ---")
print(f"json:   {time_json_loads:.4f}s")
print(f"orjson: {time_orjson_loads:.4f}s")
print(f"Speedup: {loads_speedup:.2f}x")
