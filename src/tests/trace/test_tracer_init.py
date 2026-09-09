from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from judgeval.trace.tracer import Tracer
from judgeval.trace.offline_tracer import OfflineTracer
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

    @pytest.mark.parametrize("project_id", ["", None])
    def test_neither_identifier(self, project_id):
        with patch("judgeval.trace.tracer.resolve_project_id") as resolve:
            t = Tracer.init(
                project_id=project_id,
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )
        assert t._enable_monitoring is False
        assert not t.project_id
        assert isinstance(t.get_span_exporter(), NoOpJudgmentSpanExporter)
        resolve.assert_not_called()

    @pytest.mark.parametrize(
        "missing,const",
        [
            ("api_key", "JUDGMENT_API_KEY"),
            ("organization_id", "JUDGMENT_ORG_ID"),
            ("api_url", "JUDGMENT_API_URL"),
        ],
    )
    def test_project_id_missing_credential(self, missing, const):
        kwargs = {"api_key": "k", "organization_id": "o", "api_url": "http://x"}
        kwargs[missing] = None
        with (
            patch("judgeval.trace.tracer.resolve_project_id") as resolve,
            patch(f"judgeval.trace.tracer.{const}", None),
        ):
            t = Tracer.init(project_id="pid", **kwargs)
        assert t._enable_monitoring is False
        resolve.assert_not_called()


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

    def test_project_id_only(self):
        with patch("judgeval.trace.tracer.resolve_project_id") as resolve:
            t = Tracer.init(
                project_id="supplied-pid",
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )
        assert t._enable_monitoring is True
        assert t.project_id == "supplied-pid"
        assert t.project_name is None
        assert isinstance(t.get_span_exporter(), JudgmentSpanExporter)
        resolve.assert_not_called()
        assert t._tracer_provider.resource.attributes.get("service.name") == "unknown"

    def test_project_id_wins_over_name(self):
        with patch("judgeval.trace.tracer.resolve_project_id") as resolve:
            t = Tracer.init(
                project_name="proj",
                project_id="supplied-pid",
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )
        assert t._enable_monitoring is True
        assert t.project_id == "supplied-pid"
        assert t.project_name == "proj"
        resolve.assert_not_called()
        assert t._tracer_provider.resource.attributes.get("service.name") == "proj"

    @pytest.mark.parametrize("project_id", ["", None])
    def test_empty_or_none_project_id_falls_back_to_name(self, project_id):
        with patch(
            "judgeval.trace.tracer.resolve_project_id", return_value="resolved"
        ) as resolve:
            t = Tracer.init(
                project_name="proj",
                project_id=project_id,
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )
        assert t._enable_monitoring is True
        assert t.project_id == "resolved"
        resolve.assert_called_once()
        assert resolve.call_args.args[1] == "proj"

    def test_positional_compatibility(self):
        with patch("judgeval.trace.tracer.resolve_project_id", return_value="pid"):
            t = Tracer.init("proj", "k", "o", "http://x", "staging")
        assert t.environment == "staging"
        assert (
            inspect.signature(Tracer.init).parameters["project_id"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )

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
            t._tracer_provider.resource.attributes.get("deployment.environment.name")
            == "staging"
        )
        assert "deployment.environment" not in t._tracer_provider.resource.attributes

    def test_environment_in_offline_resource(self):
        with patch(
            "judgeval.trace.offline_tracer.resolve_project_id", return_value="p"
        ):
            t = OfflineTracer.create(
                project_name="x",
                api_key="k",
                organization_id="o",
                api_url="http://api",
                environment="staging",
                set_active=False,
                dataset=[],
            )
        assert (
            t._tracer_provider.resource.attributes.get("deployment.environment.name")
            == "staging"
        )
        assert "deployment.environment" not in t._tracer_provider.resource.attributes

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
