from __future__ import annotations

from unittest.mock import patch

from judgeval.trace.tracer import Tracer
from judgeval.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)
from judgeval.trace.exporters.judgment_span_exporter import JudgmentSpanExporter
from judgeval.trace.processors.noop_judgment_span_processor import (
    NoOpJudgmentSpanProcessor,
)
from judgeval.trace.processors.judgment_span_processor import JudgmentSpanProcessor


class TestTracerInitDisabled:
    def test_missing_project_name(self):
        t = Tracer.init(api_key="k", organization_id="o", api_url="http://x")
        assert t._enable_monitoring is False
        assert isinstance(t.get_span_exporter(), NoOpJudgmentSpanExporter)
        assert isinstance(t.get_span_processor(), NoOpJudgmentSpanProcessor)

    def test_missing_api_key(self):
        t = Tracer.init(project_name="p", organization_id="o", api_url="http://x")
        assert t._enable_monitoring is False

    def test_missing_org_id(self):
        t = Tracer.init(project_name="p", api_key="k", api_url="http://x")
        assert t._enable_monitoring is False

    def test_missing_api_url(self):
        t = Tracer.init(
            project_name="p", api_key="k", organization_id="o", api_url=None
        )
        assert t._enable_monitoring is False

    def test_project_not_found(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value=None):
            t = Tracer.init(
                project_name="missing",
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )
        assert t._enable_monitoring is False


class TestTracerInitEnabled:
    def test_full_config(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value="pid"):
            t = Tracer.init(
                project_name="proj",
                api_key="key",
                organization_id="org",
                api_url="http://api/",
            )
        assert t._enable_monitoring is True
        assert t.project_id == "pid"
        assert t.project_name == "proj"
        assert isinstance(t.get_span_exporter(), JudgmentSpanExporter)
        assert isinstance(t.get_span_processor(), JudgmentSpanProcessor)

    def test_endpoint_with_trailing_slash(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api.test/",
            )
        assert (
            t.get_span_exporter()._delegate._endpoint
            == "http://api.test/otel/v1/traces"
        )

    def test_endpoint_without_trailing_slash(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api.test",
            )
        assert (
            t.get_span_exporter()._delegate._endpoint
            == "http://api.test/otel/v1/traces"
        )

    def test_environment_in_resource(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api",
                environment="staging",
            )
        assert (
            t._tracer_provider.resource.attributes.get("deployment.environment")
            == "staging"
        )

    def test_custom_resource_attributes(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api",
                resource_attributes={"custom.key": "val"},
            )
        assert t._tracer_provider.resource.attributes.get("custom.key") == "val"

    def test_exporter_cached(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x", api_key="k", organization_id="o", api_url="http://api"
            )
        assert t.get_span_exporter() is t.get_span_exporter()

    def test_processor_cached(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x", api_key="k", organization_id="o", api_url="http://api"
            )
        assert t.get_span_processor() is t.get_span_processor()
