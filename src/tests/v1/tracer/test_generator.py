import pytest
import asyncio
import contextvars
from typing import Tuple, Generator
from unittest.mock import patch, MagicMock
from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.exporters.in_memory_span_exporter import InMemorySpanExporter
from judgeval.v1.tracer.exporters.span_store import SpanStore
from judgeval.judgment_attribute_keys import AttributeKeys


@pytest.fixture
def tracer() -> Generator[Tuple[Tracer, SpanStore], None, None]:
    from opentelemetry.trace import _TRACER_PROVIDER_SET_ONCE, _TRACER_PROVIDER

    try:
        _TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        _TRACER_PROVIDER._default = None  # type: ignore[attr-defined]
    except Exception:
        pass

    mock_client = MagicMock()
    mock_client.organization_id = "test_org"
    mock_client.base_url = "http://test.com/"

    def serializer(x: object) -> str:
        return str(x)

    span_store = SpanStore()
    exporter = InMemorySpanExporter(span_store)

    with patch("judgeval.v1.utils.resolve_project_id") as mock_resolve:
        mock_resolve.return_value = "test_project_id"

        with patch.object(Tracer, "get_span_exporter", return_value=exporter):
            tracer_instance = Tracer(
                project_name="generator-test",
                enable_evaluation=False,
                enable_monitoring=True,
                api_client=mock_client,
                serializer=serializer,
                initialize=True,
            )

            yield tracer_instance, span_store

            tracer_instance.force_flush()
            tracer_instance.shutdown()


def test_sync_generator_basic(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="sync_gen")
    def sync_generator():
        yield 1
        yield 2
        yield 3

    result = list(sync_generator())
    assert result == [1, 2, 3]

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 4

    generator_span = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator"
    ][0]
    assert generator_span.name == "sync_gen"


def test_async_generator_basic(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="async_gen")
    async def async_generator():
        yield 1
        yield 2
        yield 3

    async def run_test():
        result = []
        async for item in async_generator():
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert result == [1, 2, 3]

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 4

    generator_span = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator"
    ][0]
    assert generator_span.name == "async_gen"


def test_generator_context_preservation(tracer: Tuple[Tracer, SpanStore]) -> None:
    test_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "test_var", default=None
    )
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="parent_with_context")
    def parent_function():
        test_var.set("TEST_VALUE")

        @tracer_instance.observe(span_name="gen_with_context")
        def generator_with_context():
            for i in range(3):
                assert test_var.get() == "TEST_VALUE", f"Context lost at iteration {i}"
                yield i

        return list(generator_with_context())

    result = parent_function()
    assert result == [0, 1, 2]

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 5


def test_async_generator_context_preservation(tracer: Tuple[Tracer, SpanStore]) -> None:
    test_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "test_var", default=None
    )
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="async_parent_with_context")
    async def async_parent_function():
        test_var.set("ASYNC_TEST_VALUE")

        @tracer_instance.observe(span_name="async_gen_with_context")
        async def async_generator_with_context():
            for i in range(3):
                assert test_var.get() == "ASYNC_TEST_VALUE", (
                    f"Context lost at iteration {i}"
                )
                yield i

        result = []
        async for item in async_generator_with_context():
            result.append(item)
        return result

    result = asyncio.run(async_parent_function())
    assert result == [0, 1, 2]

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 5


def test_generator_with_customer_id(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="parent_with_customer")
    def parent_with_customer():
        tracer_instance.set_customer_id("gen-customer")

        @tracer_instance.observe(span_name="child_generator")
        def child_generator():
            yield 1
            yield 2
            yield 3

        return list(child_generator())

    result = parent_with_customer()
    assert result == [1, 2, 3]

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 5

    child_spans = [
        s
        for s in spans
        if s.name == "child_generator"
        or (
            s.attributes
            and s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator_item"
        )
    ]
    for span in child_spans:
        if span.attributes:
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "gen-customer"
            )


def test_generator_exception_handling(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="failing_generator")
    def failing_generator():
        yield 1
        yield 2
        raise ValueError("Generator error")

    with pytest.raises(ValueError, match="Generator error"):
        list(failing_generator())

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 3

    generator_span = [s for s in spans if s.name == "failing_generator"][0]
    assert generator_span.name == "failing_generator"


def test_async_generator_exception_handling(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="failing_async_generator")
    async def failing_async_generator():
        yield 1
        yield 2
        raise ValueError("Async generator error")

    async def run_test():
        result = []
        async for item in failing_async_generator():
            result.append(item)

    with pytest.raises(ValueError, match="Async generator error"):
        asyncio.run(run_test())

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 3

    generator_span = [s for s in spans if s.name == "failing_async_generator"][0]
    assert generator_span.name == "failing_async_generator"


def test_generator_partial_consumption(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="partial_generator")
    def partial_generator():
        yield 1
        yield 2
        yield 3
        yield 4
        yield 5

    gen = partial_generator()
    assert next(gen) == 1
    assert next(gen) == 2
    gen.close()

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 3

    generator_span = [s for s in spans if s.name == "partial_generator"][0]
    assert generator_span.name == "partial_generator"
    assert generator_span.end_time is not None


def test_async_generator_partial_consumption(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="async_partial_generator")
    async def async_partial_generator():
        yield 1
        yield 2
        yield 3
        yield 4
        yield 5

    async def run_test():
        gen = async_partial_generator()
        assert await gen.__anext__() == 1
        assert await gen.__anext__() == 2
        await gen.aclose()

    asyncio.run(run_test())

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 3

    generator_span = [s for s in spans if s.name == "async_partial_generator"][0]
    assert generator_span.name == "async_partial_generator"
    assert generator_span.end_time is not None


def test_generator_parent_child_relationship(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="parent_function")
    def parent_function():
        @tracer_instance.observe(span_name="child_generator")
        def child_generator():
            yield 1
            yield 2

        return list(child_generator())

    parent_function()

    tracer_instance.force_flush()
    spans = span_store.get_all()
    assert len(spans) == 4

    parent_span = [s for s in spans if s.name == "parent_function"][0]
    child_gen_span = [
        s
        for s in spans
        if s.name == "child_generator"
        and s.attributes
        and s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator"
    ][0]

    assert parent_span.context is not None
    assert child_gen_span.context is not None
    assert child_gen_span.context.trace_id == parent_span.context.trace_id
    assert child_gen_span.parent is not None
    assert child_gen_span.parent.span_id == parent_span.context.span_id


def test_sync_generator_with_child_spans(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="parent_gen")
    def parent_generator():
        yield 1
        yield 2
        yield 3

    result = list(parent_generator())
    assert result == [1, 2, 3]

    tracer_instance.force_flush()
    spans = span_store.get_all()

    assert len(spans) == 4

    child_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator_item"
    ]

    assert len(child_spans) == 3

    for child_span in child_spans:
        output = (
            child_span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
            if child_span.attributes
            else None
        )
        assert output is not None


def test_async_generator_with_child_spans(tracer: Tuple[Tracer, SpanStore]) -> None:
    tracer_instance, span_store = tracer

    @tracer_instance.observe(span_name="async_parent_gen")
    async def async_parent_generator():
        yield "a"
        yield "b"
        yield "c"

    async def run_test():
        result = []
        async for item in async_parent_generator():
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert result == ["a", "b", "c"]

    tracer_instance.force_flush()
    spans = span_store.get_all()

    assert len(spans) == 4

    child_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator_item"
    ]

    assert len(child_spans) == 3

    for child_span in child_spans:
        output = (
            child_span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
            if child_span.attributes
            else None
        )
        assert output is not None
