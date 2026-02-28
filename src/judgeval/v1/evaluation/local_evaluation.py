from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, List, Tuple

from rich.console import Console
from rich.progress import Progress

from judgeval.logger import judgeval_logger
from judgeval.v1.data.example import Example
from judgeval.v1.judges import Judge
from judgeval.v1.hosted.responses import ScorerResponse
from judgeval.v1.internal.api.api_types import ExampleEvaluationRun, LocalScorerResult
from judgeval.v1.evaluation.evaluation_base import EvaluatorRunner


class LocalEvaluatorRunner(EvaluatorRunner[Judge]):
    def _build_payload(
        self,
        eval_id: str,
        project_id: str,
        eval_run_name: str,
        created_at: str,
        examples: List[Example],
        scorers: List[Judge],
    ) -> ExampleEvaluationRun:
        return {
            "id": eval_id,
            "project_id": project_id,
            "eval_name": eval_run_name,
            "created_at": created_at,
            "examples": [e.to_dict() for e in examples],
            "judgment_scorers": [],
            "custom_scorers": [],
        }

    def _submit(
        self,
        console: Console,
        project_id: str,
        eval_id: str,
        examples: List[Example],
        scorers: List[Judge],
        payload: ExampleEvaluationRun,
        progress: Progress,
    ) -> None:
        total_jobs = len(examples) * len(scorers)
        results_by_example: List[
            List[Tuple[str, ScorerResponse | None, Exception | None]]
        ] = [[] for _ in examples]

        task = progress.add_task("Running local scorers...", total=None)
        completed = 0
        start_time = time.time()

        for idx, name, result, exc in self._run_local_scorers(examples, scorers):
            results_by_example[idx].append((name, result, exc))
            completed += 1
            progress.update(
                task,
                description=f"Running local scorers... ({completed}/{total_jobs})",
            )

        elapsed = time.time() - start_time
        progress.remove_task(task)
        console.print(
            f"[green]✓[/green] Scoring completed in [bold]{elapsed:.1f}s[/bold]"
        )

        api_results: List[LocalScorerResult] = []
        for i, example in enumerate(examples):
            scorer_entries: List[Dict[str, Any]] = []
            for scorer_name, res, exc in results_by_example[i]:
                if exc is not None:
                    scorer_entries.append(
                        {
                            "scorer_name": scorer_name,
                            "value": 0,
                            "reason": "",
                            "error": str(exc),
                        }
                    )
                elif res is not None:
                    entry: Dict[str, Any] = {
                        "scorer_name": scorer_name,
                        "value": res.value,
                        "reason": res.reason,
                        "error": None,
                    }
                    if res.citations is not None:
                        entry["citations"] = res.citations
                    scorer_entries.append(entry)

            api_results.append(
                {
                    "scorers_data": scorer_entries,
                    "data_object": example.to_dict(),
                }
            )

        self._client.post_projects_eval_results_examples(
            project_id=project_id,
            payload={"results": api_results, "run": payload},
        )
        judgeval_logger.info("Local scorer results logged to backend")

    def _run_local_scorers(
        self,
        examples: List[Example],
        scorers: List[Judge],
    ) -> Generator[
        Tuple[int, str, ScorerResponse, None] | Tuple[int, str, None, Exception],
        None,
        None,
    ]:
        """Run custom scorers in a thread pool, yielding results as they complete.

        Exceptions are returned as values (like ``gather(return_exceptions=True)``).
        """

        def _run_one(
            scorer: Judge,
            example: Example,
        ) -> ScorerResponse:
            result: ScorerResponse = asyncio.run(scorer.score(example))
            return result

        with ThreadPoolExecutor() as executor:
            futures: Dict[Any, Tuple[int, str]] = {}
            for i, example in enumerate(examples):
                for scorer in scorers:
                    f = executor.submit(_run_one, scorer, example)
                    futures[f] = (i, type(scorer).__name__)

            for future in as_completed(futures):
                idx, name = futures[future]
                try:
                    res = future.result()
                except Exception as exc:
                    yield (idx, name, None, exc)
                else:
                    yield (idx, name, res, None)
