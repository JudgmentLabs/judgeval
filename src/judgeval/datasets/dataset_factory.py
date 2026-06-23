from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
)

from judgeval.exceptions import JudgmentAPIError, map_judgment_api_error
from judgeval.internal.api import JudgmentSyncClient
from judgeval.datasets.dataset import (
    Dataset,
    DatasetInfo,
    DatasetVersion,
    example_from_dataset_entry,
    example_to_dataset_entry,
)
from judgeval.data.example import Example
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id

# A column declared with a Judgment pointer type holds an id pointing at
# another resource rather than literal data. `trace` is the only one wired
# end to end today; its stored value is the trace id (a string).
_TRACE_TYPE = "trace"


DatasetColumnType = Literal[
    "string", "integer", "number", "boolean", "array", "object", "trace"
]


class DatasetSchemaProperty(TypedDict):
    """A single dataset column declaration.

    `type` is a JSON Schema primitive (`"string"`, `"integer"`,
    `"number"`, `"boolean"`, `"array"`, `"object"`) or the Judgment
    pointer type `"trace"` (value is a trace id).
    """

    type: DatasetColumnType


class DatasetSchema(TypedDict):
    """A dataset's JSON Schema.

    Datasets are object-typed with one property per column. Prefer this
    over a bare ``dict`` so editors and type checkers can catch malformed
    schemas; a plain dict of the same shape is still accepted at runtime.
    """

    type: Literal["object"]
    properties: Dict[str, DatasetSchemaProperty]


def validate_dataset_schema(schema: Mapping[str, Any]) -> None:
    """Validate a dataset JSON Schema client-side before sending.

    Mirrors the server's structural checks so obvious mistakes fail fast
    without a round-trip; the server remains the source of truth for full
    JSON Schema validation.

    Raises:
        ValueError: If the schema is not a dict, does not declare top-level
            ``type: "object"``, lacks a ``properties`` object, or declares
            more than one trace-typed column.
    """
    if not isinstance(schema, dict):
        raise ValueError("Dataset schema must be a JSON object (dict).")
    if schema.get("type") != "object":
        raise ValueError('Dataset schema must declare top-level type "object".')
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Dataset schema must declare a 'properties' object.")
    trace_cols = [
        name
        for name, prop in properties.items()
        if isinstance(prop, dict) and prop.get("type") == _TRACE_TYPE
    ]
    if len(trace_cols) > 1:
        raise ValueError(
            "A dataset may declare at most one trace column; found "
            f"{len(trace_cols)}: {', '.join(trace_cols)}."
        )


def _json_schema_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def infer_schema_from_examples(examples: Sequence[Example]) -> Dict[str, Any]:
    """Infer a JSON Schema from a set of examples.

    Convenience for `client.datasets.create()` when no explicit schema is
    supplied. Property types are inferred from the example values; every
    example must contain every declared field, so examples must share one
    shape (the same set of non-None fields). Heterogeneous examples (where
    some examples are missing fields that others have) are rejected with a
    ``ValueError``. Inferred types are JSON Schema primitives only; to
    declare a trace column (``{"type": "trace"}``) pass an explicit schema.

    Args:
        examples: Examples to infer the schema from. Must be non-empty and
            homogeneous (all examples share the same set of property keys).

    Returns:
        A JSON Schema dict of the form ``{"type": "object", "properties":
        {...}}`` declaring the example fields.

    Raises:
        ValueError: If no examples are provided or examples have
            heterogeneous fields.
    """
    if not examples:
        raise ValueError(
            "Cannot infer a dataset schema without examples. "
            "Pass an explicit `schema` to client.datasets.create()."
        )

    properties: Dict[str, Any] = {}
    # Track the full set of non-None keys seen across all examples.
    all_keys: Optional[set] = None

    for example in examples:
        keys = set()
        for key, value in example._properties.items():
            if value is None:
                continue
            keys.add(key)
            if key not in properties:
                properties[key] = {"type": _json_schema_type(value)}
        if all_keys is None:
            all_keys = keys
        elif all_keys != keys:
            # Find which fields differ between the first example's key set and
            # the current example's key set to produce a helpful message.
            extra_in_first = all_keys - keys
            extra_in_current = keys - all_keys
            missing_desc_parts = []
            if extra_in_first:
                missing_desc_parts.append(
                    f"fields present in earlier examples but missing here: "
                    f"{sorted(extra_in_first)}"
                )
            if extra_in_current:
                missing_desc_parts.append(
                    f"fields present here but missing in earlier examples: "
                    f"{sorted(extra_in_current)}"
                )
            raise ValueError(
                "All examples must share the same set of fields (dataset schemas "
                "require all declared properties). "
                + "; ".join(missing_desc_parts)
                + ". Pass an explicit `schema` or make all examples homogeneous."
            )

    return {
        "type": "object",
        "properties": properties,
    }


class DatasetFactory:
    """Create, retrieve, list, and delete datasets in your project.

    Access this via `client.datasets` -- you don't instantiate it directly.

    Datasets are schema-enforced: every dataset has a JSON Schema and all
    examples are validated against it server-side.

    Examples:
        ```python
        # Create a dataset with an explicit schema
        dataset = client.datasets.create(
            name="golden-set",
            schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "expected_output": {"type": "string"},
                },
            },
            examples=[
                Example.create(input="What is AI?", expected_output="Artificial Intelligence"),
            ],
        )

        # Retrieve an existing dataset
        dataset = client.datasets.get(name="golden-set")

        # List all datasets
        for info in client.datasets.list():
            print(info.name, info.entries)
        ```
    """

    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    def get(self, name: str) -> Optional[Dataset]:
        """Fetch an existing dataset with all its examples loaded.

        Args:
            name: The dataset name (or dataset ID).

        Returns:
            A `Dataset` with examples populated, or `None` if the project
            is not resolved.

        Examples:
            ```python
            dataset = client.datasets.get(name="golden-set")
            print(len(dataset))  # number of examples
            ```
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None
        dataset = self._client.get_projects_datasets_by_dataset_identifier(
            project_id=project_id,
            dataset_identifier=name,
        )

        dataset_kind = dataset.get("dataset_kind", "example")
        examples_data = dataset.get("examples", []) or []
        examples = [
            example_from_dataset_entry(e) for e in examples_data if isinstance(e, dict)
        ]

        current_version = dataset.get("current_version")
        judgeval_logger.info(f"Retrieved dataset {name} with {len(examples)} examples")
        return Dataset(
            name=dataset.get("name", name),
            project_id=project_id,
            dataset_id=dataset.get("dataset_id"),
            schema=dataset.get("schema"),
            current_version=(
                int(current_version) if current_version is not None else None
            ),
            dataset_kind=dataset_kind,
            examples=examples,
            client=self._client,
            project_name=self._project_name,
        )

    def create(
        self,
        name: str,
        schema: Optional[DatasetSchema] = None,
        examples: Iterable[Example] = [],
        overwrite: bool = False,
    ) -> Optional[Dataset]:
        """Create a new dataset with a JSON Schema, optionally with initial examples.

        Every dataset requires a JSON Schema (`type: "object"`); examples
        are validated against it server-side. If `schema` is omitted, a
        schema is inferred from the provided examples as a convenience --
        passing an explicit schema is recommended.

        Every example in a dataset must contain every declared schema
        field -- one shape per dataset. When inferring from examples, all
        examples must have identical non-None field sets.

        A column may be declared with `{"type": "trace"}` (any name); its
        value is a trace id rather than literal data. Trace columns must be
        declared in an explicit `schema` (inference treats values as their
        JSON primitive). At most one trace column is allowed per dataset.

        An explicit `schema` is checked client-side (`validate_dataset_schema`)
        before the request so obvious mistakes fail fast; the server performs
        the full JSON Schema validation.

        Args:
            name: Name for the dataset (unique within the project,
                case-sensitive).
            schema: JSON Schema for the dataset's examples (a `DatasetSchema`
                or a plain dict of the same shape). Required unless
                `examples` are provided to infer one from.
            examples: Examples to upload with the dataset.
            overwrite: Replace an existing dataset with the same name.
                Rejected by the server if the dataset has test configs.

        Returns:
            The new `Dataset`, or `None` if the project is not resolved.

        Raises:
            ValueError: If neither `schema` nor `examples` are provided, or
                if an explicit `schema` is structurally invalid.
            JudgmentConflictError: If a dataset with this name exists and
                `overwrite` is False.
            JudgmentValidationError: If the schema is invalid, examples
                fail validation, or overwrite is blocked by test configs.

        Examples:
            ```python
            dataset = client.datasets.create(
                name="qa-pairs",
                schema={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "expected_output": {"type": "string"},
                    },
                },
                examples=[
                    Example.create(input="What is 2+2?", expected_output="4"),
                ],
            )
            ```

            A dataset with a trace column (declared explicitly; the value
            is the trace id):

            ```python
            dataset = client.datasets.create(
                name="transcripts",
                schema={
                    "type": "object",
                    "properties": {"transcript": {"type": "trace"}},
                },
                examples=[
                    Example.create(transcript="<trace_id>"),
                ],
            )
            ```
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        if not isinstance(examples, list):
            examples = list(examples)

        resolved_schema: Dict[str, Any]
        if schema is None:
            if not examples:
                raise ValueError(
                    "Datasets require a JSON Schema. Pass `schema=...` to "
                    "client.datasets.create(), or provide `examples` to infer one."
                )
            resolved_schema = infer_schema_from_examples(examples)
            judgeval_logger.info(
                f"No schema provided for dataset {name}; inferred one from "
                f"{len(examples)} example(s)"
            )
        else:
            validate_dataset_schema(schema)
            resolved_schema = dict(schema)

        try:
            response = self._client.post_projects_datasets(
                project_id=project_id,
                payload={
                    "name": name,
                    "schema": resolved_schema,
                    "examples": [example_to_dataset_entry(e) for e in examples],
                    "dataset_kind": "example",
                    "overwrite": overwrite,
                },
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e, f"Failed to create dataset '{name}': {e.detail}"
            ) from e

        judgeval_logger.info(f"Created dataset {name}")
        created = response.get("dataset", {}) or {}
        current_version = created.get("current_version")
        return Dataset(
            name=name,
            project_id=project_id,
            dataset_id=response.get("dataset_id") or created.get("dataset_id"),
            schema=created.get("schema") or resolved_schema,
            current_version=(
                int(current_version) if current_version is not None else None
            ),
            examples=examples,
            client=self._client,
            project_name=self._project_name,
        )

    def list(self) -> Optional[List[DatasetInfo]]:
        """List all datasets in the project.

        Returns:
            A list of `DatasetInfo` summaries, or `None` if the project
            is not resolved.

        Examples:
            ```python
            for info in client.datasets.list():
                print(f"{info.name}: {info.entries} examples")
            ```
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        datasets = self._client.get_projects_datasets(
            project_id=project_id,
        )
        judgeval_logger.info(f"Fetched datasets for project {project_id}")
        return [DatasetInfo.from_dict(d) for d in datasets]

    def versions(self, name: str) -> Optional[List[DatasetVersion]]:
        """List all versions of a dataset, newest first.

        Args:
            name: The dataset name (or dataset ID).

        Returns:
            A list of `DatasetVersion` objects, or `None` if the project
            is not resolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        response = self._client.get_projects_datasets_by_dataset_identifier_versions(
            project_id=project_id,
            dataset_identifier=name,
        )
        return [DatasetVersion.from_dict(v) for v in response.get("versions", []) or []]

    def delete(self, name: str) -> bool:
        """Delete a dataset from the platform.

        Dependent test configs are deleted along with the dataset.

        Args:
            name: The dataset name (or dataset ID).

        Returns:
            True if the dataset was deleted, False if the project is not
            resolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return False

        self._client.delete_projects_datasets_by_dataset_identifier(
            project_id=project_id,
            dataset_identifier=name,
        )
        judgeval_logger.info(f"Deleted dataset {name}")
        return True
