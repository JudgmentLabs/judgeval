import threading
import time
from concurrent.futures import ThreadPoolExecutor
from judgeval.v1.trace import Tracer

fib_tracer = Tracer.init(project_name="fibonacci-threaded", set_active=False)
fizzbuzz_tracer = Tracer.init(project_name="fizzbuzz-threaded", set_active=False)


@Tracer.observe()
def fibonacci(n: int) -> int:
    time.sleep(0.05)
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@Tracer.observe()
def fizzbuzz(n: int) -> list[str]:
    time.sleep(0.05)
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


def handle_fib(n: int):
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Starting fibonacci({n})")
    fib_tracer.set_active()
    result = fibonacci(n)
    print(f"[{thread_name}] fibonacci({n}) = {result}")
    return result


def handle_fizzbuzz(n: int):
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Starting fizzbuzz({n})")
    fizzbuzz_tracer.set_active()
    result = fizzbuzz(n)
    print(f"[{thread_name}] fizzbuzz({n}) done, {len(result)} items")
    return result


def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(handle_fib, 5),
            executor.submit(handle_fizzbuzz, 10),
            executor.submit(handle_fib, 4),
            executor.submit(handle_fizzbuzz, 5),
        ]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
