import warnings
import pytest
from unittest.mock import MagicMock, patch
from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.judgment_tracer_provider import JudgmentTracerProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import context as otel_context


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def serializer():
    return lambda x: str(x)


def test_tracer_initialization(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
    )

    assert tracer.project_name == "test_project"
    assert tracer._tracer_provider is not None
    assert isinstance(tracer._tracer_provider, TracerProvider)


def test_tracer_never_sets_global_provider(mock_client, serializer):
    with patch("opentelemetry.trace.set_tracer_provider") as mock_set:
        Tracer(
            project_name="test_project",
            enable_evaluation=True,
            enable_monitoring=True,
            api_client=mock_client,
            serializer=serializer,
        )
        mock_set.assert_not_called()


def test_tracer_deprecated_isolated_param_emits_warning(mock_client, serializer):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Tracer(
            project_name="test_project",
            enable_evaluation=True,
            enable_monitoring=True,
            api_client=mock_client,
            serializer=serializer,
            isolated=False,
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "isolated" in str(w[0].message)


def test_tracer_deprecated_initialize_param_emits_warning(mock_client, serializer):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Tracer(
            project_name="test_project",
            enable_evaluation=True,
            enable_monitoring=True,
            api_client=mock_client,
            serializer=serializer,
            initialize=False,
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "initialize" in str(w[0].message)


def test_tracer_force_flush(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
    )

    tracer._tracer_provider.force_flush = MagicMock(return_value=True)
    result = tracer.force_flush(timeout_millis=5000)

    assert result is True
    tracer._tracer_provider.force_flush.assert_called_once_with(5000)


def test_tracer_shutdown(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
    )

    tracer._tracer_provider.shutdown = MagicMock()
    tracer.shutdown(timeout_millis=10000)

    tracer._tracer_provider.shutdown.assert_called_once()


def test_tracer_get_context_is_isolated(mock_client, serializer):
    tracer = Tracer(
        project_name="test_project",
        enable_evaluation=True,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=serializer,
    )

    provider = tracer.tracer_provider
    assert isinstance(provider, JudgmentTracerProvider)
    isolated_context = provider.get_isolated_current_context()
    tracer_context = tracer.get_context()

    assert tracer_context is isolated_context
    global_context = otel_context.get_current()
    assert tracer_context is not global_context


def test_multi_tracer_isolation(mock_client, serializer):
    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer_a = Tracer(
            project_name="project_a",
            enable_evaluation=False,
            enable_monitoring=False,
            api_client=mock_client,
            serializer=serializer,
        )

    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-2"):
        tracer_b = Tracer(
            project_name="project_b",
            enable_evaluation=False,
            enable_monitoring=False,
            api_client=mock_client,
            serializer=serializer,
        )

    trace_ids_a = []
    trace_ids_b = []

    with tracer_a.span("operation_a") as span_a:
        trace_ids_a.append(span_a.get_span_context().trace_id)

        with tracer_b.span("operation_b") as span_b:
            trace_ids_b.append(span_b.get_span_context().trace_id)

    assert trace_ids_a[0] != trace_ids_b[0], (
        "Different tracers should have different trace_ids"
    )


def test_multi_tracer_contexts_are_independent(mock_client, serializer):
    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer_a = Tracer(
            project_name="project_a",
            enable_evaluation=False,
            enable_monitoring=False,
            api_client=mock_client,
            serializer=serializer,
        )

    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-2"):
        tracer_b = Tracer(
            project_name="project_b",
            enable_evaluation=False,
            enable_monitoring=False,
            api_client=mock_client,
            serializer=serializer,
        )

    context_a = tracer_a.get_context()
    context_b = tracer_b.get_context()

    assert context_a is not context_b, (
        "Different tracers should have different isolated contexts"
    )


def test_multi_tracer_no_global_pollution(mock_client, serializer):
    global_context_before = otel_context.get_current()

    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer = Tracer(
            project_name="project_a",
            enable_evaluation=False,
            enable_monitoring=False,
            api_client=mock_client,
            serializer=serializer,
        )

    with tracer.span("operation"):
        global_context_during = otel_context.get_current()

    global_context_after = otel_context.get_current()

    assert global_context_before is global_context_during is global_context_after, (
        "Tracer spans should not pollute global OTel context"
    )
