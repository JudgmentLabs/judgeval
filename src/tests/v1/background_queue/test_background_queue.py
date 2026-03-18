from __future__ import annotations

import threading

import pytest

from judgeval.v1.background_queue import BackgroundQueue, enqueue, flush


@pytest.fixture(autouse=True)
def _reset_singleton():
    BackgroundQueue._instance = None
    yield
    instance = BackgroundQueue._instance
    if instance is not None:
        instance.shutdown()
    BackgroundQueue._instance = None


class TestBackgroundQueueSingleton:
    def test_get_instance_returns_same_object(self):
        a = BackgroundQueue.get_instance()
        b = BackgroundQueue.get_instance()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = BackgroundQueue.get_instance()
        BackgroundQueue._instance = None
        b = BackgroundQueue.get_instance()
        assert a is not b


class TestEnqueue:
    def test_enqueue_returns_true(self):
        q = BackgroundQueue(workers=1, max_queue_size=10)
        result = q.enqueue(lambda: None)
        assert result is True
        q.shutdown()

    def test_enqueue_executes_callable(self):
        executed = threading.Event()
        q = BackgroundQueue(workers=1, max_queue_size=10)
        q.enqueue(lambda: executed.set())
        q.force_flush()
        assert executed.is_set()
        q.shutdown()

    def test_enqueue_after_shutdown_returns_false(self):
        q = BackgroundQueue(workers=1, max_queue_size=10)
        q.shutdown()
        assert q.enqueue(lambda: None) is False

    def test_full_queue_drops_job(self):
        q = BackgroundQueue(workers=1, max_queue_size=1)
        barrier = threading.Barrier(2)

        def slow():
            barrier.wait()

        q.enqueue(slow)
        result = q.enqueue(lambda: None)
        barrier.wait()
        q.shutdown()
        assert result is False


class TestForceFlush:
    def test_force_flush_waits_for_jobs(self):
        results = []
        q = BackgroundQueue(workers=2, max_queue_size=10)
        for i in range(5):
            q.enqueue(lambda i=i: results.append(i))
        flushed = q.force_flush(timeout_ms=5000)
        assert flushed is True
        assert len(results) == 5
        q.shutdown()

    def test_force_flush_empty_queue_returns_true(self):
        q = BackgroundQueue(workers=1, max_queue_size=10)
        assert q.force_flush() is True
        q.shutdown()

    def test_force_flush_after_shutdown_returns_false(self):
        q = BackgroundQueue(workers=1, max_queue_size=10)
        q.shutdown()
        assert q.force_flush() is False


class TestModuleLevelFunctions:
    def test_module_enqueue(self):
        executed = threading.Event()
        enqueue(lambda: executed.set())
        flush(timeout_ms=5000)
        assert executed.is_set()

    def test_module_flush(self):
        results = []
        for i in range(3):
            enqueue(lambda i=i: results.append(i))
        ok = flush(timeout_ms=5000)
        assert ok is True
        assert len(results) == 3
