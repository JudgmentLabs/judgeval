from __future__ import annotations

import datetime
import orjson
import os
import yaml
from dataclasses import dataclass
from typing import List, Literal, Optional, Iterable, Iterator
from itertools import islice
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)


from judgeval.v1.data.example import Example
from judgeval.v1.internal.api import JudgmentSyncClient
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


@dataclass
class DatasetInfo:
    dataset_id: str
    name: str
    created_at: str
    kind: str
    entries: float
    creator: str


@dataclass
class Dataset:
    name: str
    project_id: str
    project_name: str  # Display name from Judgeval client
    dataset_kind: str = "example"
    examples: Optional[List[Example]] = None
    client: Optional[JudgmentSyncClient] = None

    def add_from_json(self, file_path: str, batch_size: int = 100) -> None:
        with open(file_path, "rb") as file:
            data = orjson.loads(file.read())
        examples = []
        for e in data:
            if isinstance(e, dict):
                name = e.get("name")
                example = Example(name=name)
                for key, value in e.items():
                    if key != "name":
                        example.set_property(key, value)
                examples.append(example)
            else:
                examples.append(e)
        self.add_examples(examples, batch_size=batch_size)

    def add_from_yaml(self, file_path: str, batch_size: int = 100) -> None:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        examples = []
        for e in data:
            if isinstance(e, dict):
                name = e.get("name")
                example = Example(name=name)
                for key, value in e.items():
                    if key != "name":
                        example.set_property(key, value)
                examples.append(example)
            else:
                examples.append(e)
        self.add_examples(examples, batch_size=batch_size)

    def add_examples(self, examples: Iterable[Example], batch_size: int = 100) -> None:
        if not self.client:
            return

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

                self.client.post_projects_datasets_by_dataset_name_examples(
                    project_id=self.project_id,
                    dataset_name=self.name,
                    payload={"examples": [e.to_dict() for e in batch]},
                )

        judgeval_logger.info(
            f"Successfully added {total_uploaded} examples to dataset {self.name}"
        )

    def save_as(
        self,
        file_type: Literal["json", "yaml"],
        dir_path: str,
        save_name: Optional[str] = None,
    ) -> None:
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
                    value = str(example.get_property(key) or "")
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
