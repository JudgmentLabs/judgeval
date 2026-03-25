from judgeval import Tracer

tracer_a = Tracer.init(project_name="my-project")


@Tracer.observe()
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    result = add(1, 2)
    print(f"Result: {result}")
