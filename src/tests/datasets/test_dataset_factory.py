from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from judgeval.data.example import Example
from judgeval.data.trace import TraceRef
from judgeval.datasets.dataset import Dataset, DatasetInfo, DatasetVersion
from judgeval.datasets.dataset_factory import (
    DatasetFactory,
    infer_schema_from_examples,
)
from judgeval.exceptions import (
    JudgmentAPIError,
    JudgmentConflictError,
    JudgmentValidationError,
)


def _make_factory(project_id="proj-1"):
    client = MagicMock()
    return DatasetFactory(
        client=client, project_id=project_id, project_name="test-project"
    ), client


class TestDatasetFactoryGet:
    def test_get_returns_dataset(self):
        factory, client = _make_factory()
        client.get_projects_datasets_by_dataset_identifier.return_value = {
            "dataset_id": "d1",
            "name": "my-dataset",
            "dataset_kind": "example",
            "schema": {"type": "object", "properties": {}},
            "current_version": 1,
            "examples": [],
        }
        ds = factory.get("my-dataset")
        assert isinstance(ds, Dataset)
        assert ds.name == "my-dataset"
        assert ds.dataset_id == "d1"
        assert ds.schema == {"type": "object", "properties": {}}
        assert ds.current_version == 1

    def test_get_maps_nested_examples(self):
        factory, client = _make_factory()
        client.get_projects_datasets_by_dataset_identifier.return_value = {
            "dataset_id": "d1",
            "name": "ds",
            "dataset_kind": "example",
            "examples": [
                {
                    "example_id": "e1",
                    "created_at": "2024-01-01",
                    "data": {"input": "q"},
                    "offline_trace_id": None,
                }
            ],
        }
        ds = factory.get("ds")
        assert len(ds) == 1
        assert ds.examples[0]["input"] == "q"
        assert ds.examples[0].example_id == "e1"

    def test_get_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.get("ds")
        assert result is None


class TestDatasetFactoryCreate:
    def test_create_with_schema(self):
        factory, client = _make_factory()
        schema = {
            "type": "object",
            "properties": {"input": {"type": "string"}},
        }
        client.post_projects_datasets.return_value = {
            "detail": "ok",
            "dataset_id": "d1",
            "dataset": {"dataset_id": "d1", "name": "new-ds", "current_version": 1},
        }
        ds = factory.create("new-ds", schema=schema)
        assert isinstance(ds, Dataset)
        assert ds.name == "new-ds"
        assert ds.dataset_id == "d1"
        payload = client.post_projects_datasets.call_args.kwargs["payload"]
        assert payload["schema"] == schema
        assert payload["name"] == "new-ds"
        assert payload["overwrite"] is False

    def test_create_sends_examples_inline(self):
        factory, client = _make_factory()
        client.post_projects_datasets.return_value = {
            "detail": "ok",
            "dataset_id": "d1",
            "dataset": {},
        }
        examples = [Example.create(input="q", output="a")]
        factory.create(
            "new-ds",
            schema={"type": "object", "properties": {}},
            examples=examples,
        )
        payload = client.post_projects_datasets.call_args.kwargs["payload"]
        assert len(payload["examples"]) == 1
        assert payload["examples"][0]["input"] == "q"
        assert payload["examples"][0]["example_id"] == examples[0].example_id

    def test_create_without_schema_or_examples_raises(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError, match="JSON Schema"):
            factory.create("new-ds")

    def test_create_infers_schema_from_examples(self):
        factory, client = _make_factory()
        client.post_projects_datasets.return_value = {
            "detail": "ok",
            "dataset_id": "d1",
            "dataset": {},
        }
        factory.create("new-ds", examples=[Example.create(input="q", score=1.5)])
        payload = client.post_projects_datasets.call_args.kwargs["payload"]
        assert payload["schema"]["type"] == "object"
        assert payload["schema"]["properties"]["input"] == {"type": "string"}
        assert payload["schema"]["properties"]["score"] == {"type": "number"}
        # Inferred schemas must not contain a "required" key.
        assert "required" not in payload["schema"]

    def test_create_conflict_maps_to_conflict_error(self):
        factory, client = _make_factory()
        client.post_projects_datasets.side_effect = JudgmentAPIError(
            409, "a dataset with this name already exists", None
        )
        with pytest.raises(JudgmentConflictError):
            factory.create("dupe", schema={"type": "object", "properties": {}})

    def test_create_validation_maps_to_validation_error(self):
        factory, client = _make_factory()
        client.post_projects_datasets.side_effect = JudgmentAPIError(
            422, "examples failed schema validation", None
        )
        with pytest.raises(JudgmentValidationError):
            factory.create("bad", schema={"type": "object", "properties": {}})

    def test_create_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.create("ds", schema={"type": "object", "properties": {}})
        assert result is None


class TestDatasetFactoryList:
    def test_list_returns_dataset_infos(self):
        factory, client = _make_factory()
        client.get_projects_datasets.return_value = [
            {
                "dataset_id": "d1",
                "name": "ds1",
                "created_at": "2024-01-01",
                "entries": 5,
                "current_version": 2,
                "test_config_count": 1,
                "creator_id": "user-1",
                "schema": {"type": "object", "properties": {}},
            }
        ]
        result = factory.list()
        assert len(result) == 1
        assert isinstance(result[0], DatasetInfo)
        assert result[0].name == "ds1"
        assert result[0].entries == 5
        assert result[0].test_config_count == 1

    def test_list_tolerates_missing_fields(self):
        factory, client = _make_factory()
        client.get_projects_datasets.return_value = [{"dataset_id": "d1", "name": "x"}]
        result = factory.list()
        assert result[0].entries is None

    def test_list_missing_project_id_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        result = factory.list()
        assert result is None


class TestDatasetFactoryVersionsAndDelete:
    def test_versions(self):
        factory, client = _make_factory()
        client.get_projects_datasets_by_dataset_identifier_versions.return_value = {
            "versions": [{"version_id": "v1", "dataset_id": "d1", "version_number": 1}]
        }
        versions = factory.versions("ds")
        assert len(versions) == 1
        assert isinstance(versions[0], DatasetVersion)

    def test_delete(self):
        factory, client = _make_factory()
        assert factory.delete("ds") is True
        call = client.delete_projects_datasets_by_dataset_identifier.call_args
        assert call.kwargs["dataset_identifier"] == "ds"

    def test_delete_missing_project_id_returns_false(self):
        factory, _ = _make_factory(project_id=None)
        assert factory.delete("ds") is False


class TestInferSchema:
    def test_infers_types(self):
        schema = infer_schema_from_examples(
            [
                Example.create(
                    input="q",
                    count=3,
                    score=0.5,
                    flag=True,
                    items=[1],
                    nested={"a": 1},
                )
            ]
        )
        props = schema["properties"]
        assert props["input"] == {"type": "string"}
        assert props["count"] == {"type": "integer"}
        assert props["score"] == {"type": "number"}
        assert props["flag"] == {"type": "boolean"}
        assert props["items"] == {"type": "array"}
        assert props["nested"] == {"type": "object"}

    def test_no_required_key_in_schema(self):
        schema = infer_schema_from_examples(
            [
                Example.create(input="q", expected_output="a"),
                Example.create(input="q2", expected_output="b"),
            ]
        )
        assert "required" not in schema
        assert "expected_output" in schema["properties"]
        assert "input" in schema["properties"]

    def test_heterogeneous_fields_raises(self):
        with pytest.raises(ValueError, match="same set of fields"):
            infer_schema_from_examples(
                [
                    Example.create(input="q", expected_output="a"),
                    Example.create(input="q2"),
                ]
            )

    def test_heterogeneous_fields_error_names_missing_fields(self):
        with pytest.raises(ValueError, match="expected_output"):
            infer_schema_from_examples(
                [
                    Example.create(input="q", expected_output="a"),
                    Example.create(input="q2"),
                ]
            )

    def test_traceref_infers_trace_typed_column_under_any_name(self):
        schema = infer_schema_from_examples(
            [Example.create(question="q", transcript=TraceRef("t1"))]
        )
        assert schema["properties"]["transcript"] == {"type": "trace"}
        assert schema["properties"]["question"] == {"type": "string"}

    def test_all_traceref_examples_emit_trace_typed_column(self):
        schema = infer_schema_from_examples(
            [
                Example.create(transcript=TraceRef("t1")),
                Example.create(transcript=TraceRef("t2")),
            ]
        )
        assert schema["properties"]["transcript"] == {"type": "trace"}
        assert "required" not in schema

    def test_more_than_one_trace_column_raises(self):
        with pytest.raises(ValueError, match="at most one trace column"):
            infer_schema_from_examples(
                [
                    Example.create(trace=TraceRef("t1"), apple=TraceRef("t2")),
                ]
            )

    def test_untraced_examples_omit_trace_column(self):
        schema = infer_schema_from_examples(
            [Example.create(input="q1"), Example.create(input="q2")]
        )
        assert all(
            prop.get("type") != "trace" for prop in schema["properties"].values()
        )
        assert "required" not in schema

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            infer_schema_from_examples([])
