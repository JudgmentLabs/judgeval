from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from fireworks import Dataset  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from judgeval.v1.trainers.config import TrainerConfig, ModelConfig
    from judgeval.v1.trainers.trainable_model import TrainableModel
    from judgeval.v1.tracer.tracer import Tracer
    from judgeval.v1.scorers.base_scorer import BaseScorer
    from judgeval.v1.internal.api import JudgmentSyncClient

from judgeval.v1.trainers.base_trainer import BaseTrainer
from judgeval.v1.tracer.exporters import SpanStore, InMemorySpanExporter
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.data.example import Example
from judgeval.v1.data.scoring_result import ScoringResult
from judgeval.v1.internal.api.api_types import ExampleEvaluationRun
from judgeval.v1.trainers.console import (
    _spinner_progress,
    _print_progress,
    _print_progress_update,
)
from judgeval.exceptions import JudgmentRuntimeError
from opentelemetry import trace


class FireworksTrainer(BaseTrainer):
    __slots__ = ("_client", "span_store", "span_exporter")

    def __init__(
        self,
        config: "TrainerConfig",
        trainable_model: "TrainableModel",
        tracer: "Tracer",
        project_name: Optional[str] = None,
        client: Optional["JudgmentSyncClient"] = None,
    ):
        super().__init__(config, trainable_model, tracer, project_name)
        if client is None:
            raise ValueError("client is required")
        self._client = client
        self.span_store = SpanStore()
        self.span_exporter = InMemorySpanExporter(self.span_store)

    def _extract_message_history_from_spans(
        self, trace_id: str
    ) -> List[Dict[str, str]]:
        spans = self.span_store.get_by_trace_id(trace_id)
        if not spans:
            return []

        messages = []
        first_found = False

        for span in sorted(spans, key=lambda s: getattr(s, "start_time", 0)):
            span_attributes = span.attributes or {}
            span_type = span_attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND, "span")

            if (
                not span_attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
                and span_type != "llm"
            ):
                continue

            if span_type == "llm":
                if not first_found and span_attributes.get(
                    AttributeKeys.JUDGMENT_INPUT
                ):
                    input_data: Any = span_attributes.get(
                        AttributeKeys.JUDGMENT_INPUT, {}
                    )
                    if isinstance(input_data, dict) and "messages" in input_data:
                        input_messages = input_data["messages"]
                        if input_messages:
                            first_found = True
                            for msg in input_messages:
                                if (
                                    isinstance(msg, dict)
                                    and "role" in msg
                                    and "content" in msg
                                ):
                                    messages.append(
                                        {"role": msg["role"], "content": msg["content"]}
                                    )

                output = span_attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
                if output is not None:
                    content = str(output)
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "messages" in parsed:
                            for msg in parsed["messages"]:
                                if (
                                    isinstance(msg, dict)
                                    and msg.get("role") == "assistant"
                                ):
                                    content = msg.get("content", content)
                                    break
                    except (json.JSONDecodeError, KeyError):
                        pass
                    messages.append({"role": "assistant", "content": content})

            elif span_type in ("user", "tool"):
                output = span_attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
                if output is not None:
                    content = str(output)
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "messages" in parsed:
                            for msg in parsed["messages"]:
                                if isinstance(msg, dict) and msg.get("role") == "user":
                                    content = msg.get("content", content)
                                    break
                    except (json.JSONDecodeError, KeyError):
                        pass
                    messages.append({"role": "user", "content": content})

        return messages

    async def generate_rollouts_and_rewards(
        self,
        agent_function: Callable[..., Any],
        scorers: List["BaseScorer"],
        prompts: dict[int, dict[Any, Any]],
        num_prompts_per_step: Optional[int] = None,
        num_generations_per_prompt: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        num_prompts_per_step = min(
            num_prompts_per_step or self.config.num_prompts_per_step, len(prompts)
        )
        num_generations_per_prompt = (
            num_generations_per_prompt or self.config.num_generations_per_prompt
        )
        concurrency = concurrency or self.config.concurrency

        semaphore = asyncio.Semaphore(concurrency)

        @self.tracer.observe(span_type="function")
        async def generate_single_response(prompt_id: int, generation_id: int):
            async with semaphore:
                prompt_input = prompts[prompt_id]
                response_data = await agent_function(**prompt_input)
                messages = response_data.get("messages", [])

                current_span = trace.get_current_span()
                trace_id = None
                if current_span and current_span.is_recording():
                    trace_id = format(current_span.get_span_context().trace_id, "032x")

                try:
                    if trace_id is not None:
                        traced_messages = self._extract_message_history_from_spans(
                            trace_id
                        )
                        if traced_messages:
                            messages = traced_messages
                except Exception as e:
                    print(f"Warning: Failed to get message history from trace: {e}")
                finally:
                    if trace_id is not None:
                        self.span_store.clear_trace(trace_id)

                example = Example()
                example.set_property("input", prompt_input)
                example.set_property("messages", messages)
                example.set_property("actual_output", response_data)

                evaluation_run: ExampleEvaluationRun = {
                    "project_name": self.project_name,
                    "eval_name": f"training_step_{self.trainable_model.current_step}_prompt_{prompt_id}_gen_{generation_id}",
                    "examples": [example.to_dict()],
                    "judgment_scorers": [
                        scorer.get_scorer_config() for scorer in scorers
                    ],
                }

                response = self._client.add_to_run_eval_queue_examples(evaluation_run)
                if not response.get("success", False):
                    raise JudgmentRuntimeError(
                        f"Failed to queue evaluation: {response.get('error', 'Unknown error')}"
                    )

                results = await self._poll_evaluation_until_complete(
                    evaluation_run["eval_name"], len(scorers)
                )

                if results and results[0].scorers_data:
                    scores = [
                        scorer_data.score
                        for scorer_data in results[0].scorers_data
                        if scorer_data.score is not None
                    ]
                    reward = sum(scores) / len(scores) if scores else 0.0
                else:
                    reward = 0.0

            return {
                "prompt_id": prompt_id,
                "generation_id": generation_id,
                "messages": messages,
                "evals": {"score": reward},
            }

        coros = []
        for prompt_id in range(num_prompts_per_step):
            for generation_id in range(num_generations_per_prompt):
                coro = generate_single_response(prompt_id, generation_id)
                coros.append(coro)

        with _spinner_progress(f"Generating {len(coros)} rollouts..."):
            num_completed = 0
            results = []

            for future in asyncio.as_completed(coros):
                result = await future
                results.append(result)
                num_completed += 1

        _print_progress(f"Generated {len(results)} rollouts successfully")

        dataset_rows = []
        for prompt_id in range(num_prompts_per_step):
            prompt_generations = [r for r in results if r["prompt_id"] == prompt_id]
            sample_generations = [
                {"messages": gen["messages"], "evals": gen["evals"]}
                for gen in prompt_generations
            ]
            dataset_rows.append({"samples": sample_generations})

        return dataset_rows

    async def _poll_evaluation_until_complete(
        self, eval_name: str, expected_scorers: int
    ) -> List[ScoringResult]:
        import time

        max_wait_time = 300
        poll_interval = 2
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            await asyncio.sleep(poll_interval)

            from judgeval.v1.internal.api.api_types import EvalResultsFetch

            fetch_request: EvalResultsFetch = {
                "experiment_run_id": eval_name,
                "project_name": self.project_name,
            }

            try:
                response = self._client.fetch_experiment_run(fetch_request)
                if response and response.get("results"):
                    results_data = response.get("results")
                    if results_data is not None and len(results_data) > 0:
                        scoring_results = []
                        for result_data in results_data:
                            from judgeval.v1.data.scorer_data import ScorerData

                            scorers_data = []
                            for scorer_result in result_data.get("scorers_data", []):
                                scorers_data.append(
                                    ScorerData(
                                        name=scorer_result.get("name", ""),
                                        threshold=scorer_result.get("threshold", 0.0),
                                        success=scorer_result.get("success", False),
                                        score=scorer_result.get("score"),
                                        reason=scorer_result.get("reason"),
                                    )
                                )

                            scoring_results.append(
                                ScoringResult(
                                    success=result_data.get("success", False),
                                    scorers_data=scorers_data,
                                    name=result_data.get("name"),
                                    trace_id=result_data.get("trace_id"),
                                    run_duration=result_data.get("run_duration"),
                                    evaluation_cost=result_data.get("evaluation_cost"),
                                )
                            )
                        return scoring_results
            except Exception:
                pass

        raise JudgmentRuntimeError(
            f"Evaluation {eval_name} did not complete within {max_wait_time} seconds"
        )

    async def run_reinforcement_learning(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List["BaseScorer"],
        prompts: dict[int, dict[Any, Any]],
    ) -> "ModelConfig":
        _print_progress("Starting reinforcement learning training")

        training_params = {
            "num_steps": self.config.num_steps,
            "num_prompts_per_step": self.config.num_prompts_per_step,
            "num_generations_per_prompt": self.config.num_generations_per_prompt,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        start_step = self.trainable_model.current_step

        for step in range(start_step, self.config.num_steps):
            step_num = step + 1
            _print_progress(
                f"Starting training step {step_num}", step_num, self.config.num_steps
            )

            self.trainable_model.advance_to_next_step(step)

            dataset_rows = await self.generate_rollouts_and_rewards(
                agent_function, scorers, prompts
            )

            with _spinner_progress(
                "Preparing training dataset", step_num, self.config.num_steps
            ):
                dataset = Dataset.from_list(dataset_rows)
                dataset.sync()

            _print_progress(
                "Starting reinforcement training", step_num, self.config.num_steps
            )
            job = self.trainable_model.perform_reinforcement_step(dataset, step)

            last_state = None
            with _spinner_progress(
                "Training job in progress", step_num, self.config.num_steps
            ):
                while not job.is_completed:
                    job.raise_if_bad_state()
                    current_state = job.state

                    if current_state != last_state:
                        if current_state in ["uploading", "validating"]:
                            _print_progress_update(
                                f"Training job: {current_state} data"
                            )
                        elif current_state == "training":
                            _print_progress_update(
                                "Training job: model training in progress"
                            )
                        else:
                            _print_progress_update(f"Training job: {current_state}")
                        last_state = current_state

                    await asyncio.sleep(10)
                    job = job.get()
                    if job is None:
                        raise JudgmentRuntimeError(
                            "Training job was deleted while waiting for completion"
                        )

            _print_progress(
                f"Training completed! New model: {job.output_model}",
                step_num,
                self.config.num_steps,
            )

        _print_progress("All training steps completed!")

        with _spinner_progress("Deploying final trained model"):
            self.trainable_model.advance_to_next_step(self.config.num_steps)

        return self.trainable_model.get_model_config(training_params)

    async def train(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List["BaseScorer"],
        prompts: dict[int, dict[Any, Any]],
    ) -> "ModelConfig":
        try:
            return await self.run_reinforcement_learning(
                agent_function, scorers, prompts
            )
        except JudgmentRuntimeError:
            raise
        except Exception as e:
            raise JudgmentRuntimeError(f"Training process failed: {str(e)}") from e
