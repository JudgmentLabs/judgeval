"""
Concurrency tests for the v1 SDK.

These tests verify that the SDK is thread-safe and handles concurrent
access correctly without race conditions or data corruption.

Run with: pytest src/tests/reliability/test_concurrency.py -v
"""

import pytest
import threading
import asyncio
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set
from unittest.mock import patch

from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.tracer_factory import TracerFactory


@pytest.mark.reliability
class TestThreadSafety:
    """Test that SDK operations are thread-safe."""

    def test_concurrent_observe_no_race_conditions(self, tracer: Tracer):
        """
        100 threads making 100 calls each should not cause race conditions.

        All calls should complete successfully with correct results.
        """
        THREADS = 100
        CALLS_PER_THREAD = 100

        results: List[str] = []
        errors: List[Exception] = []
        results_lock = threading.Lock()

        @tracer.observe(span_type="function")
        def traced_function(thread_id: int, call_id: int) -> str:
            time.sleep(random.uniform(0.0001, 0.001))  # Small random delay
            return f"thread-{thread_id}-call-{call_id}"

        def worker(thread_id: int):
            for call_id in range(CALLS_PER_THREAD):
                try:
                    result = traced_function(thread_id, call_id)
                    with results_lock:
                        results.append(result)
                except Exception as e:
                    with results_lock:
                        errors.append(e)

        threads = []
        for i in range(THREADS):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        expected_count = THREADS * CALLS_PER_THREAD

        assert len(errors) == 0, f"Got {len(errors)} errors: {errors[:5]}"
        assert len(results) == expected_count, (
            f"Expected {expected_count} results, got {len(results)}"
        )

    def test_concurrent_span_creation(self, tracer: Tracer):
        """
        Creating spans concurrently should be safe.
        """
        THREADS = 50
        SPANS_PER_THREAD = 50

        success_count = 0
        error_count = 0
        count_lock = threading.Lock()

        def worker(thread_id: int):
            nonlocal success_count, error_count
            for i in range(SPANS_PER_THREAD):
                try:
                    with tracer.span(f"thread-{thread_id}-span-{i}"):
                        tracer.set_attribute("thread_id", thread_id)
                        tracer.set_attribute("span_index", i)
                        time.sleep(random.uniform(0.0001, 0.0005))
                    with count_lock:
                        success_count += 1
                except Exception:
                    with count_lock:
                        error_count += 1

        threads = []
        for i in range(THREADS):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        expected = THREADS * SPANS_PER_THREAD

        assert error_count == 0, (
            f"Got {error_count} errors during concurrent span creation"
        )
        assert success_count == expected, (
            f"Expected {expected} spans, got {success_count}"
        )

    def test_context_isolation_between_threads(self, tracer: Tracer):
        """
        Each thread should have isolated span context.

        Setting attributes in one thread should not affect another.
        """
        THREADS = 20

        thread_results: dict = {}
        results_lock = threading.Lock()

        def worker(thread_id: int):
            with tracer.span(f"thread-{thread_id}-span"):
                # Set thread-specific attribute
                tracer.set_attribute("thread_id", thread_id)

                # Simulate some work
                time.sleep(random.uniform(0.01, 0.05))

                # The attribute should still be this thread's value
                # (We can't easily read it back, so we just verify no crashes)

                with results_lock:
                    thread_results[thread_id] = "completed"

        threads = []
        for i in range(THREADS):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(thread_results) == THREADS, (
            f"Only {len(thread_results)} of {THREADS} threads completed"
        )

    def test_thread_pool_executor_compatibility(self, tracer: Tracer):
        """
        SDK should work correctly with ThreadPoolExecutor.
        """
        TASKS = 500
        WORKERS = 20

        @tracer.observe(span_type="function")
        def task_function(task_id: int) -> int:
            time.sleep(random.uniform(0.001, 0.005))
            return task_id * 2

        results = []

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(task_function, i) for i in range(TASKS)]
            for future in as_completed(futures):
                results.append(future.result())

        assert len(results) == TASKS
        assert sum(results) == sum(i * 2 for i in range(TASKS))


@pytest.mark.reliability
class TestAsyncConcurrency:
    """Test async concurrency handling."""

    @pytest.mark.asyncio
    async def test_async_concurrent_observe(self, tracer: Tracer):
        """
        Concurrent async traced functions should work correctly.
        """
        TASKS = 100

        @tracer.observe(span_type="function")
        async def async_traced_function(task_id: int) -> str:
            await asyncio.sleep(random.uniform(0.001, 0.01))
            return f"result-{task_id}"

        tasks = [async_traced_function(i) for i in range(TASKS)]
        results = await asyncio.gather(*tasks)

        assert len(results) == TASKS
        assert all(f"result-{i}" in results for i in range(TASKS))

    @pytest.mark.asyncio
    async def test_async_span_context_isolation(self, tracer: Tracer):
        """
        Each async task should have isolated span context.
        """
        TASKS = 50
        results: List[int] = []

        async def task_with_span(task_id: int):
            with tracer.span(f"async-task-{task_id}"):
                tracer.set_attribute("task_id", task_id)
                await asyncio.sleep(random.uniform(0.001, 0.01))
                results.append(task_id)

        await asyncio.gather(*[task_with_span(i) for i in range(TASKS)])

        assert len(results) == TASKS
        assert set(results) == set(range(TASKS))

    @pytest.mark.asyncio
    async def test_mixed_sync_async_tracing(self, tracer: Tracer):
        """
        Mixing sync and async traced functions should work.
        """
        ITERATIONS = 50

        @tracer.observe(span_type="function")
        def sync_function(i: int) -> str:
            return f"sync-{i}"

        @tracer.observe(span_type="function")
        async def async_function(i: int) -> str:
            await asyncio.sleep(0.001)
            return f"async-{i}"

        async def mixed_workflow(i: int) -> tuple:
            sync_result = sync_function(i)
            async_result = await async_function(i)
            return (sync_result, async_result)

        tasks = [mixed_workflow(i) for i in range(ITERATIONS)]
        results = await asyncio.gather(*tasks)

        assert len(results) == ITERATIONS
        for i, (sync_res, async_res) in enumerate(results):
            assert sync_res == f"sync-{i}"
            assert async_res == f"async-{i}"


@pytest.mark.reliability
class TestNestedConcurrency:
    """Test nested concurrent operations."""

    def test_nested_threads_with_tracing(self, tracer: Tracer):
        """
        Nested thread creation with tracing should work.
        """
        OUTER_THREADS = 10
        INNER_THREADS = 5

        results: List[str] = []
        results_lock = threading.Lock()

        @tracer.observe(span_type="function")
        def inner_task(outer_id: int, inner_id: int) -> str:
            time.sleep(random.uniform(0.001, 0.005))
            return f"outer-{outer_id}-inner-{inner_id}"

        def outer_worker(outer_id: int):
            with tracer.span(f"outer-{outer_id}"):
                inner_threads = []
                for inner_id in range(INNER_THREADS):

                    def inner_worker(oid=outer_id, iid=inner_id):
                        result = inner_task(oid, iid)
                        with results_lock:
                            results.append(result)

                    t = threading.Thread(target=inner_worker)
                    inner_threads.append(t)
                    t.start()

                for t in inner_threads:
                    t.join()

        outer_threads = []
        for i in range(OUTER_THREADS):
            t = threading.Thread(target=outer_worker, args=(i,))
            outer_threads.append(t)
            t.start()

        for t in outer_threads:
            t.join()

        expected = OUTER_THREADS * INNER_THREADS
        assert len(results) == expected, (
            f"Expected {expected} results, got {len(results)}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_tasks_with_nested_spans(self, tracer: Tracer):
        """
        Concurrent async tasks with nested spans should work correctly.
        """
        TASKS = 20
        NESTING_DEPTH = 5

        completed_tasks: Set[int] = set()

        async def nested_span_task(task_id: int, depth: int = 0):
            with tracer.span(f"task-{task_id}-depth-{depth}"):
                tracer.set_attribute("task_id", task_id)
                tracer.set_attribute("depth", depth)

                if depth < NESTING_DEPTH:
                    await asyncio.sleep(random.uniform(0.001, 0.005))
                    await nested_span_task(task_id, depth + 1)
                else:
                    completed_tasks.add(task_id)

        await asyncio.gather(*[nested_span_task(i) for i in range(TASKS)])

        assert len(completed_tasks) == TASKS


@pytest.mark.reliability
class TestMultipleTracersConcurrency:
    """Test concurrent use of multiple tracer instances."""

    def test_multiple_tracers_concurrent_use(self, mock_client):
        """
        Multiple tracer instances used concurrently should not interfere.
        """
        TRACERS = 5
        CALLS_PER_TRACER = 50

        tracers = []
        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            for i in range(TRACERS):
                factory = TracerFactory(mock_client)
                tracer = factory.create(
                    project_name=f"tracer-{i}",
                    enable_monitoring=True,
                    enable_evaluation=False,
                    isolated=True,
                )
                tracers.append(tracer)

        results: List[tuple] = []
        results_lock = threading.Lock()

        def worker(tracer_idx: int):
            tracer = tracers[tracer_idx]

            @tracer.observe(span_type="function")
            def traced_function(call_id: int):
                return f"tracer-{tracer_idx}-call-{call_id}"

            for call_id in range(CALLS_PER_TRACER):
                result = traced_function(call_id)
                with results_lock:
                    results.append((tracer_idx, result))

        threads = []
        for i in range(TRACERS):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        expected = TRACERS * CALLS_PER_TRACER
        assert len(results) == expected

        # Verify results are correct for each tracer
        for tracer_idx, result in results:
            assert result.startswith(f"tracer-{tracer_idx}-")
