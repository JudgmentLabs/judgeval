from __future__ import annotations

from unittest.mock import patch

import pytest

from judgeval.judgeval import Judgeval
from judgeval.evaluation.evaluation_factory import EvaluationFactory
from judgeval.datasets.dataset_factory import DatasetFactory
from judgeval.prompts.prompt_factory import PromptFactory


class TestJudgevalInit:
    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            with patch.dict("os.environ", {}, clear=True):
                with patch("judgeval.judgeval.JUDGMENT_API_KEY", None):
                    Judgeval(
                        project_name="p",
                        api_key=None,
                        organization_id="o",
                        api_url="http://x",
                    )

    def test_missing_org_id_raises(self):
        with pytest.raises(ValueError, match="organization_id"):
            with patch("judgeval.judgeval.JUDGMENT_ORG_ID", None):
                Judgeval(
                    project_name="p",
                    api_key="k",
                    organization_id=None,
                    api_url="http://x",
                )

    def test_missing_api_url_raises(self):
        with pytest.raises(ValueError, match="api_url"):
            with patch("judgeval.judgeval.JUDGMENT_API_URL", None):
                Judgeval(
                    project_name="p",
                    api_key="k",
                    organization_id="o",
                    api_url=None,
                )

    def test_missing_project_name_raises(self):
        with pytest.raises(ValueError, match="project_name"):
            Judgeval(
                project_name="",
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )

    def test_valid_init(self):
        with patch("judgeval.judgeval.resolve_project_id", return_value="proj-1"):
            j = Judgeval(
                project_name="my-project",
                api_key="key",
                organization_id="org",
                api_url="http://api",
            )
        assert j._project_name == "my-project"
        assert j._project_id == "proj-1"

    def test_project_not_found_warns_but_does_not_raise(self):
        with patch("judgeval.judgeval.resolve_project_id", return_value=None):
            j = Judgeval(
                project_name="missing",
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )
        assert j._project_id is None


class TestJudgevalProperties:
    def setup_method(self):
        with patch("judgeval.judgeval.resolve_project_id", return_value="p-1"):
            self.j = Judgeval(
                project_name="proj",
                api_key="k",
                organization_id="o",
                api_url="http://x",
            )

    def test_evaluation_property_returns_factory(self):
        assert isinstance(self.j.evaluation, EvaluationFactory)

    def test_datasets_property_returns_factory(self):
        assert isinstance(self.j.datasets, DatasetFactory)

    def test_prompts_property_returns_factory(self):
        assert isinstance(self.j.prompts, PromptFactory)
