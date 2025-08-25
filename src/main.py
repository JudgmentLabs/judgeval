from judgeval.tracer import Tracer


tracer1 = Tracer(project_name="errors")
tracer2 = Tracer(project_name="errors")


@tracer1.observe(span_type="function")
def foo():
    return "Hello world! - Foo"


@tracer2.observe(span_type="function")
def bar():
    return "Hello world! - Bar"


@tracer1.observe(span_type="function")
@tracer2.observe(span_type="function")
def both():
    foo()
    bar()


if __name__ == "__main__":
    both()
    tracer1.force_flush()
    tracer2.force_flush()
