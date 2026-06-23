from __future__ import annotations

import datetime
import orjson
import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Iterable, Iterator
from itertools import islice
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)


from judgeval.data.example import Example
from judgeval.exceptions import JudgmentAPIError, map_judgment_api_error
from judgeval.internal.api import JudgmentSyncClient
from judgeval.logger import judgeval_logger


def _batch_examples(
    examples: Iterable[Example], batch_size: int = 100
) -> Iterator[List[Example]]:
    """Generator that yields batches of examples for efficient memory usage.

    Works with any iterable including generators, consuming only batch_size items at a time.
    """
    iterator = iter(examples)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def example_to_dataset_entry(example: Example) -> Dict[str, Any]:
    """Serialize an `Example` into the dataset example payload shape.

    The example's custom properties become the example data fields;
    `example_id` and `created_at` are lifted to the top level. For a
    trace-typed column (declared `{"type": "trace"}` in the dataset
    schema), set the field to the trace id string.
    """
    entry: Dict[str, Any] = {
        "example_id": example.example_id,
        "created_at": example.created_at,
    }
    entry.update(example._properties)
    return entry


def example_from_dataset_entry(entry: Dict[str, Any]) -> Example:
    """Build an `Example` from a server dataset example payload.

    The server returns examples as `{example_id, created_at, data,
    offline_trace_id, metadata, ...}` with the example fields nested
    under `data`.
    """
    data = entry.get("data")
    if not isinstance(data, dict):
        data = {
            k: v
            for k, v in entry.items()
            if k not in ("example_id", "created_at", "name")
        }

    example = Example(
        example_id=entry.get("example_id", "") or "",
        created_at=entry.get("created_at", "") or "",
        name=entry.get("name"),
    )
    for key, value in data.items():
        example._properties[key] = value

    offline_trace_id = entry.get("offline_trace_id")
    if offline_trace_id and "offline_trace_id" not in example._properties:
        example._properties["offline_trace_id"] = offline_trace_id
    return example


@dataclass
class DatasetInfo:
    """Summary metadata returned when listing datasets.

    Returned by `client.datasets.list()`. Use the `name` to fetch the
    full dataset with `client.datasets.get(name=...)`.

    Attributes:
        dataset_id: Unique dataset identifier.
        name: Dataset name.
        created_at: ISO-8601 creation timestamp.
        entries: Number of examples in the dataset.
        current_version: Latest dataset version number.
        test_config_count: Number of test configs referencing the dataset.
        creator_id: ID of the user who created the dataset.
        schema: The dataset's JSON Schema.
    """

    dataset_id: str
    name: str
    created_at: Optional[str] = None
    entries: Optional[float] = None
    current_version: Optional[float] = None
    test_config_count: Optional[float] = None
    creator_id: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetInfo:
        return cls(
            dataset_id=data.get("dataset_id", ""),
            name=data.get("name", ""),
            created_at=data.get("created_at"),
            entries=data.get("entries"),
            current_version=data.get("current_version"),
            test_config_count=data.get("test_config_count"),
            creator_id=data.get("creator_id"),
            schema=data.get("schema"),
        )


@dataclass
class DatasetVersion:
    """A single immutable version of a dataset.

    Returned by `Dataset.versions()` / `client.datasets.versions()`.

    Attributes:
        version_id: Unique version identifier.
        dataset_id: Owning dataset ID.
        version_number: Monotonically increasing version number.
        created_at: ISO-8601 creation timestamp.
        item_count: Number of examples in this version.
        user_id: ID of the user who created the version.
    """

    version_id: str
    dataset_id: str
    version_number: int
    created_at: Optional[str] = None
    item_count: Optional[int] = None
    user_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetVersion:
        return cls(
            version_id=data.get("version_id", ""),
            dataset_id=data.get("dataset_id", ""),
            version_number=int(data.get("version_number", 0) or 0),
            created_at=data.get("created_at"),
            item_count=(
                int(data["item_count"]) if data.get("item_count") is not None else None
            ),
            user_id=data.get("user_id"),
        )


@dataclass
class Dataset:
    """A schema-enforced collection of `Example` objects on the Judgment platform.

    Datasets are created and retrieved via `client.datasets`. Every
    dataset has a JSON Schema that all examples are validated against
    server-side. Once you have a `Dataset`, you can append examples, add
    trace-backed examples, list versions, iterate over examples, export
    to JSON/YAML, or display a rich table preview.

    Attributes:
        name: Dataset name.
        project_id: Owning project ID.
        project_name: Project name.
        dataset_id: Unique dataset identifier (set when created/fetched).
        schema: The dataset's JSON Schema. Examples must conform to it.
        current_version: Latest dataset version number.
        dataset_kind: Kind of dataset (default `"example"`).
        examples: The loaded examples (populated when using `.get()`).
        client: Internal API client (set automatically).

    Examples:
        Create a dataset and add examples:

        ```python
        dataset = client.datasets.create(
            name="golden-set",
            schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "expected_output": {"type": "string"},
                },
            },
        )
        dataset.add_examples([
            Example.create(input="What is AI?", expected_output="Artificial Intelligence"),
        ])
        ```

        Retrieve and iterate:

        ```python
        dataset = client.datasets.get(name="golden-set")
        for example in dataset:
            print(example["input"])
        ```
    """

    name: str
    project_id: str
    project_name: str
    dataset_id: Optional[str] = None
    schema: Optional[Dict[str, Any]] = field(default=None)
    current_version: Optional[int] = None
    dataset_kind: str = "example"
    examples: Optional[List[Example]] = None
    client: Optional[JudgmentSyncClient] = None

    @property
    def _identifier(self) -> str:
        return self.dataset_id or self.name

    def add_from_json(self, file_path: str, batch_size: int = 100) -> None:
        """Upload examples from a JSON file.

        The file should contain a JSON array of objects, where each object
        has the fields you want as example properties (e.g. `input`,
        `actual_output`, `expected_output`).

        Args:
            file_path: Path to the JSON file.
            batch_size: Number of examples uploaded per API call.

        Examples:
            ```python
            dataset.add_from_json("./data/golden-set.json")
            ```
        """
        with open(file_path, "rb") as file:
            data = orjson.loads(file.read())
        examples = []
        for e in data:
            if isinstance(e, dict):
                name = e.get("name")
                example = Example(name=name)
                for key, value in e.items():
                    if key != "name":
                        example._properties[key] = value
                examples.append(example)
            else:
                examples.append(e)
        self.add_examples(examples, batch_size=batch_size)

    def add_from_yaml(self, file_path: str, batch_size: int = 100) -> None:
        """Upload examples from a YAML file.

        Same as `add_from_json` but reads YAML format.

        Args:
            file_path: Path to the YAML file.
            batch_size: Number of examples uploaded per API call.
        """
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        examples = []
        for e in data:
            if isinstance(e, dict):
                name = e.get("name")
                example = Example(name=name)
                for key, value in e.items():
                    if key != "name":
                        example._properties[key] = value
                examples.append(example)
            else:
                examples.append(e)
        self.add_examples(examples, batch_size=batch_size)

    def add_examples(self, examples: Iterable[Example], batch_size: int = 100) -> None:
        """Append `Example` objects to this dataset.

        Examples are validated server-side against the dataset schema and
        uploaded in batches with a progress bar. Each successful batch
        advances the dataset version. Accepts any iterable, including
        generators.

        Args:
            examples: A list (or iterable) of `Example` objects.
            batch_size: Number of examples per upload batch.

        Raises:
            TypeError: If a single `Example` is passed instead of a list.
            JudgmentValidationError: If examples fail schema validation.
        """
        if not self.client:
            return

        if isinstance(examples, Example):
            raise TypeError(
                "examples must be a list of Example objects, not a single Example. "
                "Use add_examples([example]) instead."
            )

        batches = _batch_examples(examples, batch_size)
        total_uploaded = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(pulse_style="green"),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[info]}"),
        ) as progress:
            task = progress.add_task(
                f"Uploading to {self.name}",
                total=None,
                info="",
            )

            batch_num = 0
            for batch in batches:
                if len(batch) > 0 and not isinstance(batch[0], Example):
                    raise TypeError("Examples must be a list of Example objects")

                batch_num += 1
                batch_size_actual = len(batch)
                total_uploaded += batch_size_actual

                progress.update(
                    task,
                    advance=1,
                    info=f"Batch {batch_num} ({batch_size_actual} examples, {total_uploaded} total)",
                )

                try:
                    response = self.client.post_projects_datasets_by_dataset_identifier_examples(
                        project_id=self.project_id,
                        dataset_identifier=self._identifier,
                        payload={
                            "examples": [example_to_dataset_entry(e) for e in batch]
                        },
                    )
                except JudgmentAPIError as e:
                    raise map_judgment_api_error(
                        e,
                        f"Failed to add examples to dataset '{self.name}': {e.detail}",
                    ) from e

                version_added = response.get("version_added")
                if version_added is not None:
                    self.current_version = int(version_added)

        judgeval_logger.info(
            f"Successfully added {total_uploaded} examples to dataset {self.name}"
        )

    def versions(self) -> List[DatasetVersion]:
        """List all versions of this dataset, newest first.

        Returns:
            A list of `DatasetVersion` objects.
        """
        if not self.client:
            return []

        response = self.client.get_projects_datasets_by_dataset_identifier_versions(
            project_id=self.project_id,
            dataset_identifier=self._identifier,
        )
        return [DatasetVersion.from_dict(v) for v in response.get("versions", []) or []]

    def delete(self) -> None:
        """Delete this dataset from the platform.

        Dependent test configs are deleted along with the dataset.
        """
        if not self.client:
            return

        self.client.delete_projects_datasets_by_dataset_identifier(
            project_id=self.project_id,
            dataset_identifier=self._identifier,
        )
        judgeval_logger.info(f"Deleted dataset {self.name}")

    def save_as(
        self,
        file_type: Literal["json", "yaml"],
        dir_path: str,
        save_name: Optional[str] = None,
    ) -> None:
        """Export the dataset to a local JSON or YAML file.

        Args:
            file_type: `"json"` or `"yaml"`.
            dir_path: Directory to write into (created if it doesn't exist).
            save_name: File name without extension. Defaults to a timestamp.

        Examples:
            ```python
            dataset = client.datasets.get(name="golden-set")
            dataset.save_as("json", dir_path="./exports")
            ```
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = save_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        complete_path = os.path.join(dir_path, f"{file_name}.{file_type}")

        examples_data = [e.to_dict() for e in self.examples] if self.examples else []

        if file_type == "json":
            with open(complete_path, "wb") as file:
                file.write(
                    orjson.dumps(
                        {"examples": examples_data}, option=orjson.OPT_INDENT_2
                    )
                )
        elif file_type == "yaml":
            with open(complete_path, "w") as file:
                yaml.dump({"examples": examples_data}, file, default_flow_style=False)

    def __iter__(self):
        return iter(self.examples or [])

    def __len__(self):
        return len(self.examples) if self.examples else 0

    def __str__(self):
        return f"Dataset(name={self.name}, examples={len(self.examples) if self.examples else 0})"

    def display(self, max_examples: int = 5) -> None:
        """Print a formatted table preview to the terminal.

        Args:
            max_examples: Maximum number of examples to show.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        total = len(self.examples) if self.examples else 0
        console.print(f"\n[bold cyan]Dataset: {self.name}[/bold cyan]")
        console.print(f"[dim]Project:[/dim] {self.project_name}")
        console.print(f"[dim]Total examples:[/dim] {total}")

        if not self.examples:
            console.print("[dim]No examples found[/dim]")
            return

        display_count = min(max_examples, total)

        if total > 0:
            first_example = self.examples[0]
            property_keys = list(first_example.properties.keys())

            table = Table(show_header=True, header_style="bold")
            table.add_column("#", style="dim", width=4)
            table.add_column("Name", style="cyan")
            for key in property_keys[:3]:
                table.add_column(key, max_width=30)

            for i, example in enumerate(self.examples[:display_count]):
                row = [str(i + 1), example.name or "—"]
                for key in property_keys[:3]:
                    value = str(example._properties.get(key) or "")
                    if len(value) > 30:
                        value = value[:27] + "..."
                    row.append(value)
                table.add_row(*row)

            console.print()
            console.print(table)

            if total > display_count:
                console.print(
                    f"[dim]... and {total - display_count} more examples[/dim]"
                )

        console.print()
