import time
from judgeval import Judgeval

judgeval = Judgeval(project_name="test")
tracer = judgeval.tracer.create()


@tracer.observe()
def fib(n):
    time.sleep(0.1)
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


print(fib(5))
