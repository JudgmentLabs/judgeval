from __future__ import annotations

import threading
from typing import Any

import pytest
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value

import judgeval.trajectory.batch_observation_processor as batch_processor
from judgeval.trajectory.batch_observation_processor import (
    BatchObservationProcessor,
)


class RecordingExporter:
    """Records each exported batch and whether instrumentation was off."""

    def __init__(self) -> None:
        self.batches: list[list[Any]] = []
        self.suppressed: list[bool] = []
        self.release = threading.Event()
        self.release.set()

    def export(self, batch: list[Any]) -> None:
        self.release.wait(5)
        self.suppressed.append(bool(get_value(_SUPPRESS_INSTRUMENTATION_KEY)))
        self.batches.append(batch)


@pytest.fixture
def processors():
    created: list[BatchObservationProcessor] = []

    def make(exporter: RecordingExporter) -> BatchObservationProcessor:
        processor = BatchObservationProcessor(exporter)  # type: ignore[arg-type]
        created.append(processor)
        return processor

    yield make
    for processor in created:
        processor.shutdown(timeout_seconds=1)


def item(index: int) -> Any:
    return {"observation": {"conversation_id": str(index)}, "observed_at": "t"}


def test_flush_exports_everything_in_batches_of_at_most_one_hundred(processors):
    exporter = RecordingExporter()
    processor = processors(exporter)
    for index in range(250):
        processor.emit(item(index))
    assert processor.force_flush()
    assert sum(len(batch) for batch in exporter.batches) == 250
    assert max(len(batch) for batch in exporter.batches) <= 100


def test_exports_with_instrumentation_suppressed(processors):
    exporter = RecordingExporter()
    processor = processors(exporter)
    processor.emit(item(0))
    processor.force_flush()
    assert exporter.suppressed == [True]


def test_a_full_queue_drops_its_oldest_observations(processors):
    exporter = RecordingExporter()
    processor = processors(exporter)
    capacity = batch_processor.MAX_QUEUE_SIZE
    # Holding the export lock keeps the worker from draining the queue.
    with processor._export_lock:
        for index in range(capacity + 1):
            processor.emit(item(index))
        assert (len(processor._queue), processor._queue[0]) == (capacity, item(1))


def test_shutdown_exports_the_queue_and_ignores_later_observations(processors):
    exporter = RecordingExporter()
    processor = processors(exporter)
    processor.emit(item(0))
    processor.shutdown(timeout_seconds=5)
    processor.emit(item(1))
    assert (exporter.batches, processor.force_flush()) == ([[item(0)]], False)


def test_a_forked_child_starts_with_an_empty_queue_and_its_own_worker(processors):
    exporter = RecordingExporter()
    processor = processors(exporter)
    parent_worker = processor._worker_thread
    processor._queue.append(item(0))
    processor._start()
    assert (len(processor._queue), processor._worker_thread is parent_worker) == (
        0,
        False,
    )
