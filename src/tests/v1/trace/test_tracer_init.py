"""Tests for Tracer.init() — configuration validation and exporter/processor wiring."""

from __future__ import annotations

from unittest.mock import patch

from judgeval.v1.trace.tracer import Tracer
from judgeval.v1.trace.exporters.noop_span_exporter import NoOpSpanExporter
from judgeval.v1.trace.exporters.judgment_span_exporter import JudgmentSpanExporter
from judgeval.v1.trace.processors.noop_span_processor import NoOpSpanProcessor
from judgeval.v1.trace.processors.judgment_span_processor import JudgmentSpanProcessor


class TestTracerInitValidation:
    def test_missing_project_name_disables_monitoring(self):
        t = Tracer.init(
            project_name=None,
            api_key="k",
            organization_id="o",
            api_url="http://x",
        )
        assert t._enable_monitoring is False
        assert isinstance(t.get_span_exporter(), NoOpSpanExporter)
        assert isinstance(t.get_span_processor(), NoOpSpanProcessor)

    def test_missing_api_key_disables_monitoring(self):
        t = Tracer.init(
            project_name="p",
            api_key=None,
            organization_id="o",
            api_url="http://x",
        )
        assert t._enable_monitoring is False

    def test_missing_org_id_disables_monitoring(self):
        t = Tracer.init(
            project_name="p",
            api_key="k",
            organization_id=None,
            api_url="http://x",
        )
        assert t._enable_monitoring is False

    def test_missing_api_url_disables_monitoring(self):
        t = Tracer.init(
            project_name="p",
            api_key="k",
            organization_id="o",
            api_url=None,
        )
        assert t._enable_monitoring is False

    def test_project_not_found_disables_monitoring(self):
        with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value=None):
            t = Tracer.init(
                project_name="missing",
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )
        assert t._enable_monitoring is False


class TestTracerInitSuccess:
    def test_full_config_enables_monitoring(self):
        with patch(
            "judgeval.v1.trace.tracer.resolve_project_id", return_value="proj-abc"
        ):
            t = Tracer.init(
                project_name="my-project",
                api_key="key",
                organization_id="org",
                api_url="http://api.test/",
            )
        assert t._enable_monitoring is True
        assert t.project_id == "proj-abc"
        assert t.project_name == "my-project"
        assert isinstance(t.get_span_exporter(), JudgmentSpanExporter)
        assert isinstance(t.get_span_processor(), JudgmentSpanProcessor)

    def test_exporter_endpoint_with_trailing_slash(self):
        with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api.test/",
            )
        exporter = t.get_span_exporter()
        # The delegate OTLPSpanExporter was created with the correct endpoint
        assert exporter._delegate._endpoint == "http://api.test/otel/v1/traces"

    def test_exporter_endpoint_without_trailing_slash(self):
        with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api.test",
            )
        exporter = t.get_span_exporter()
        assert exporter._delegate._endpoint == "http://api.test/otel/v1/traces"

    def test_environment_added_to_resource(self):
        with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api.test",
                environment="staging",
            )
        resource = t._tracer_provider.resource
        assert resource.attributes.get("deployment.environment") == "staging"

    def test_custom_resource_attributes(self):
        with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api.test",
                resource_attributes={"custom.key": "custom.value"},
            )
        resource = t._tracer_provider.resource
        assert resource.attributes.get("custom.key") == "custom.value"

    def test_exporter_cached_on_second_call(self):
        with patch("judgeval.v1.trace.tracer.resolve_project_id", return_value="p"):
            t = Tracer.init(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api.test",
            )
        e1 = t.get_span_exporter()
        e2 = t.get_span_exporter()
        assert e1 is e2
