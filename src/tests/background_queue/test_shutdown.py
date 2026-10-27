from __future__ import annotations

import atexit
import threading
import time

import pytest
from judgeval.background_queue import BackgroundQueue
import judgeval.background_queue as queue_module


def test_shutdown_drains_accepted_work_before_returning():
    queue = BackgroundQueue(workers=1, max_queue_size=2)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    returned = threading.Event()
    finished_at_return = []

    # --- Controlled scheduling helpers ---
    def job():
        started.set()
        assert release.wait(5)
        finished.set()

    def shutdown():
        queue.shutdown(timeout_ms=3000)
        finished_at_return.append(finished.is_set())
        returned.set()

    # --- End controlled scheduling helpers ---
    assert queue.enqueue(job)
    assert started.wait(5)
    thread = threading.Thread(target=shutdown)
    thread.start()
    try:
        # Let an early return expose itself before releasing the job. A
        # draining shutdown stays blocked until the job can complete.
        returned.wait(0.1)
        release.set()
        thread.join(5)
        assert returned.is_set()
        assert finished_at_return == [True], "shutdown returned before accepted work"
        assert finished.is_set()
        assert queue.enqueue(lambda: None) is False
    finally:
        release.set()
        thread.join(5)
        queue.shutdown()


def test_shutdown_racing_with_admission_never_raises(monkeypatch):
    queue = BackgroundQueue(workers=1, max_queue_size=1)
    submitting = threading.Event()
    shutdown_started = threading.Event()
    executor_closed = threading.Event()
    ran = threading.Event()
    outcomes = []
    original_submit = queue._executor.submit
    original_shutdown = queue._executor.shutdown

    # --- Controlled scheduling helpers ---
    def submit(*args, **kwargs):
        submitting.set()
        assert shutdown_started.wait(5)
        # On the broken queue, shutdown closes the executor while enqueue
        # is paused here. A serialized queue keeps shutdown out until submit
        # finishes; the timeout releases this artificial scheduling pause.
        executor_closed.wait(0.2)
        return original_submit(*args, **kwargs)

    def close_executor(*args, **kwargs):
        result = original_shutdown(*args, **kwargs)
        executor_closed.set()
        return result

    def enqueue():
        try:
            outcomes.append(queue.enqueue(ran.set))
        except Exception as error:
            outcomes.append(error)

    def shutdown():
        shutdown_started.set()
        queue.shutdown(timeout_ms=3000)

    # --- End controlled scheduling helpers ---
    monkeypatch.setattr(queue._executor, "submit", submit)
    monkeypatch.setattr(queue._executor, "shutdown", close_executor)
    producer = threading.Thread(target=enqueue)
    closer = threading.Thread(target=shutdown)
    producer.start()
    assert submitting.wait(5)
    closer.start()
    producer.join(5)
    closer.join(5)
    assert not producer.is_alive()
    assert not closer.is_alive()
    assert outcomes == [True]
    assert ran.is_set()
    assert queue.enqueue(lambda: None) is False


def test_shutdown_timeout_leaves_accepted_jobs_runnable():
    queue = BackgroundQueue(workers=1, max_queue_size=2)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    queued_finished = threading.Event()

    # --- Controlled scheduling helpers ---
    def job():
        started.set()
        assert release.wait(5)
        finished.set()

    # --- End controlled scheduling helpers ---
    try:
        assert queue.enqueue(job)
        assert started.wait(5)
        assert queue.enqueue(queued_finished.set)
        start = time.monotonic()
        queue.shutdown(timeout_ms=0)
        assert time.monotonic() - start < 1
        assert not finished.is_set()
        assert not queued_finished.is_set()
        assert queue.enqueue(lambda: None) is False
        release.set()
        assert finished.wait(5)
        assert queued_finished.wait(5)
        queue.shutdown()
    finally:
        release.set()
        queue.shutdown(timeout_ms=0)


def test_completed_future_callback_does_not_deadlock(monkeypatch):
    queue = BackgroundQueue(workers=1, max_queue_size=1)
    original_submit = queue._executor.submit
    outcomes = []

    # --- Controlled scheduling helpers ---
    def submit(*args, **kwargs):
        future = original_submit(*args, **kwargs)
        future.result(timeout=5)
        return future

    def enqueue_twice():
        outcomes.append(queue.enqueue(lambda: None))
        outcomes.append(queue.enqueue(lambda: None))

    # --- End controlled scheduling helpers ---
    monkeypatch.setattr(queue._executor, "submit", submit)
    producer = threading.Thread(target=enqueue_twice, daemon=True)
    producer.start()
    producer.join(5)
    if producer.is_alive():
        # A regression must fail the test instead of hanging at interpreter exit.
        atexit.unregister(queue.shutdown)
        queue._executor.shutdown(wait=False)
        pytest.fail("completed callback deadlocked with admission")
    assert outcomes == [True, True]
    queue.shutdown()


def test_submit_failure_restores_capacity(monkeypatch):
    queue = BackgroundQueue(workers=1, max_queue_size=1)
    ran = threading.Event()
    original_submit = queue._executor.submit

    # --- Failure injection helper ---
    def fail_submit(*args, **kwargs):
        raise RuntimeError("injected submission failure")

    # --- End failure injection helper ---
    try:
        with monkeypatch.context() as patch:
            patch.setattr(queue._executor, "submit", fail_submit)
            with pytest.raises(RuntimeError, match="injected submission failure"):
                queue.enqueue(lambda: None)
        assert queue._executor.submit == original_submit
        assert queue.enqueue(ran.set)
        assert queue.force_flush()
        assert ran.is_set()
    finally:
        queue.shutdown()


def test_force_flush_waits_on_snapshot_without_blocking_admission(monkeypatch):
    queue = BackgroundQueue(workers=2, max_queue_size=2)
    first_release = threading.Event()
    second_release = threading.Event()
    waiting = threading.Event()
    outcomes = []
    original_wait = queue_module.wait

    # --- Controlled scheduling helper ---
    def tracked_wait(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)

    # --- End controlled scheduling helper ---
    monkeypatch.setattr(queue_module, "wait", tracked_wait)
    assert queue.enqueue(lambda: first_release.wait(5))
    thread = threading.Thread(target=lambda: outcomes.append(queue.force_flush()))
    thread.start()
    try:
        assert waiting.wait(5)
        assert queue.enqueue(lambda: second_release.wait(5))
        first_release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert outcomes == [True]
        assert not second_release.is_set()
    finally:
        first_release.set()
        second_release.set()
        thread.join(5)
        queue.shutdown()


def test_failed_job_releases_capacity(monkeypatch):
    queue = BackgroundQueue(workers=1, max_queue_size=1)
    logged = threading.Event()
    ran = threading.Event()

    # --- Failure injection helper ---
    def fail_job():
        raise ValueError("job failed")

    # --- End failure injection helper ---
    monkeypatch.setattr(queue_module.judgeval_logger, "error", lambda _: logged.set())
    try:
        assert queue.enqueue(fail_job)
        assert logged.wait(5)
        assert queue.enqueue(ran.set)
        assert queue.force_flush()
        assert ran.is_set()
    finally:
        queue.shutdown()


@pytest.mark.parametrize("from_callback", [False, True], ids=["job", "callback"])
def test_worker_shutdown_leaves_queued_work_runnable(
    monkeypatch, caplog, from_callback
):
    queue = BackgroundQueue(workers=1, max_queue_size=2)
    started = threading.Event()
    release = threading.Event()
    shutdown_returned = threading.Event()
    queued_finished = threading.Event()
    outcomes = []

    # --- Controlled scheduling helpers ---
    def shutdown():
        queue.shutdown(timeout_ms=100)
        outcomes.append("shutdown returned")
        shutdown_returned.set()

    def job():
        started.set()
        assert release.wait(5)
        if from_callback:
            raise ValueError("trigger completion callback")
        shutdown()

    def queued_job():
        outcomes.append("queued job ran")
        queued_finished.set()

    # --- End controlled scheduling helpers ---
    if from_callback:
        monkeypatch.setattr(queue_module.judgeval_logger, "error", lambda _: shutdown())
    try:
        assert queue.enqueue(job)
        assert started.wait(5)
        assert queue.enqueue(queued_job)
        release.set()
        assert shutdown_returned.wait(5)
        assert queued_finished.wait(5)
        assert outcomes == ["shutdown returned", "queued job ran"]
        assert queue.enqueue(lambda: None) is False
        assert "Flush timed out" not in caplog.text
    finally:
        release.set()
        queue.shutdown(timeout_ms=0)


def test_worker_of_another_queue_still_drains_external_shutdown():
    caller = BackgroundQueue(workers=1, max_queue_size=1)
    target = BackgroundQueue(workers=1, max_queue_size=1)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    returned = threading.Event()
    finished_at_return = []

    # --- Controlled scheduling helpers ---
    def target_job():
        started.set()
        assert release.wait(5)
        finished.set()

    def shutdown_target():
        target.shutdown(timeout_ms=3000)
        finished_at_return.append(finished.is_set())
        returned.set()

    # --- End controlled scheduling helpers ---
    try:
        assert target.enqueue(target_job)
        assert started.wait(5)
        assert caller.enqueue(shutdown_target)
        returned.wait(0.1)
        release.set()
        assert returned.wait(5)
        assert finished_at_return == [True]
    finally:
        release.set()
        caller.shutdown()
        target.shutdown()
