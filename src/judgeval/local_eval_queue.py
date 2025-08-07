"""Local evaluation queue for batching custom scorer evaluations.

This module provides a simple in-memory queue for EvaluationRun objects that contain
only local (BaseScorer) scorers. Useful for batching evaluations and processing them
either synchronously or in a background thread.
"""

import queue
import threading
from typing import Callable, List, Optional
import time

from judgeval.common.logger import judgeval_logger
from judgeval.constants import MAX_CONCURRENT_EVALUATIONS
from judgeval.data import ScoringResult
from judgeval.data.evaluation_run import EvaluationRun
from judgeval.utils.async_utils import safe_run_async
from judgeval.scorers.score import a_execute_scoring


class LocalEvaluationQueue:
    """Lightweight in-memory queue for local evaluation runs.

    Only supports EvaluationRuns with local scorers (BaseScorer instances).
    API scorers (APIScorerConfig) are not supported as they have their own queue.
    """

    def __init__(
        self, max_concurrent: int = MAX_CONCURRENT_EVALUATIONS, num_workers: int = 4
    ):
        if num_workers <= 0:
            raise ValueError("num_workers must be a positive integer.")
        self._queue: queue.Queue[Optional[EvaluationRun]] = queue.Queue()
        self._max_concurrent = max_concurrent
        self._num_workers = num_workers  # Number of worker threads
        self._worker_threads: List[threading.Thread] = []
        self._shutdown_event = threading.Event()

    def enqueue(self, evaluation_run: EvaluationRun) -> None:
        """Add evaluation run to the queue."""
        self._queue.put(evaluation_run)

    def _process_run(self, evaluation_run: EvaluationRun) -> List[ScoringResult]:
        """Execute evaluation run locally and return results."""

        if not evaluation_run.custom_scorers:
            raise ValueError(
                "LocalEvaluationQueue only supports runs with local scorers (BaseScorer). "
                "Found only APIScorerConfig instances."
            )

        return safe_run_async(
            a_execute_scoring(
                evaluation_run.examples,
                evaluation_run.custom_scorers,
                model=evaluation_run.model,
                throttle_value=0,
                max_concurrent=self._max_concurrent // self._num_workers,
                show_progress=False,
            )
        )

    def run_all(
        self,
        callback: Optional[Callable[[EvaluationRun, List[ScoringResult]], None]] = None,
    ) -> None:
        """Process all queued runs synchronously.

        Args:
            callback: Optional function called after each run with (run, results).
        """
        while not self._queue.empty():
            run = self._queue.get()
            if run is None:  # Sentinel for worker shutdown
                self._queue.put(None)
                break
            results = self._process_run(run)
            if callback:
                callback(run, results)
            self._queue.task_done()

    def start_workers(
        self,
        callback: Optional[Callable[[EvaluationRun, List[ScoringResult]], None]] = None,
    ) -> List[threading.Thread]:
        """Start multiple background threads to process runs in parallel.

        Args:
            callback: Optional function called after each run with (run, results).

        Returns:
            List of started worker threads.
        """

        def _worker(worker_id: int) -> None:
            while not self._shutdown_event.is_set():
                try:
                    # Use timeout so workers can check shutdown event periodically
                    run = self._queue.get(timeout=1.0)
                    if run is None:  # Sentinel to stop worker
                        # Put sentinel back for other workers
                        self._queue.put(None)
                        self._queue.task_done()
                        break

                    try:
                        results = self._process_run(run)
                        if callback:
                            callback(run, results)
                    except Exception as exc:
                        judgeval_logger.error(
                            f"Worker {worker_id} error processing {run.eval_name}: {exc}"
                        )
                        # Continue processing other runs instead of shutting down all workers
                    finally:
                        self._queue.task_done()

                except queue.Empty:
                    # Timeout - check shutdown event and continue
                    continue

        # Start worker threads
        for i in range(self._num_workers):
            thread = threading.Thread(target=_worker, args=(i,), daemon=True)
            thread.start()
            self._worker_threads.append(thread)

        return self._worker_threads

    def start_worker(
        self,
        callback: Optional[Callable[[EvaluationRun, List[ScoringResult]], None]] = None,
    ) -> Optional[threading.Thread]:
        """Start a single background thread to process runs (backward compatibility).

        Args:
            callback: Optional function called after each run with (run, results).

        Returns:
            The started thread, or None if no threads were started.
        """
        threads = self.start_workers(callback)
        return threads[0] if threads else None

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all queued tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            True if all tasks completed, False if timeout occurred.
        """
        try:
            if timeout is None:
                self._queue.join()
                return True
            else:
                start_time = time.time()
                while not self._queue.empty() or self._queue.unfinished_tasks > 0:
                    if time.time() - start_time > timeout:
                        return False
                    time.sleep(0.1)
                return True
        except Exception:
            return False

    def stop_workers(self) -> None:
        """Signal all background workers to stop after current tasks complete."""
        if not self._worker_threads:
            return

        # Signal shutdown
        self._shutdown_event.set()

        # Send sentinel to wake up any blocking workers
        for _ in range(self._num_workers):
            self._queue.put(None)

        # Wait for all workers to finish with timeout
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    judgeval_logger.warning(
                        f"Worker thread {thread.name} did not shut down gracefully"
                    )

        self._worker_threads.clear()
        self._shutdown_event.clear()
