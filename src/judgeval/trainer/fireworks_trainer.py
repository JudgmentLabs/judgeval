import asyncio
import json
from typing import Optional, Callable, Any, List, Union, Dict
from fireworks import Dataset  # type: ignore[import-not-found]
from .config import TrainerConfig, ModelConfig
from .base_trainer import BaseTrainer
from .trainable_model import TrainableModel
from judgeval.tracer import Tracer
from judgeval.tracer.exporters.store import SpanStore
from judgeval.tracer.exporters import InMemorySpanExporter
from judgeval.tracer.keys import AttributeKeys
from judgeval import JudgmentClient
from judgeval.scorers import ExampleScorer, ExampleAPIScorerConfig
from judgeval.data import Example
from .console import _spinner_progress, _print_progress, _print_progress_update
from judgeval.exceptions import JudgmentRuntimeError


class FireworksTrainer(BaseTrainer):
    """
    Fireworks AI implementation of the training provider.

    This trainer uses Fireworks AI's infrastructure for reinforcement learning
    fine-tuning (RFT) of language models.
    """

    def __init__(
        self,
        config: TrainerConfig,
        trainable_model: TrainableModel,
        tracer: Tracer,
        project_name: Optional[str] = None,
    ):
        """
        Initialize the FireworksTrainer.

        Args:
            config: TrainerConfig instance with training parameters
            trainable_model: TrainableModel instance for Fireworks training
            tracer: Tracer for observability
            project_name: Project name for organizing training runs and evaluations
        """
        try:
            super().__init__(config, trainable_model, tracer, project_name)

            self.judgment_client = JudgmentClient()
            self.span_store = SpanStore()
            self.span_exporter = InMemorySpanExporter(self.span_store)
        except Exception as e:
            raise JudgmentRuntimeError(
                f"Failed to initialize FireworksTrainer: {str(e)}"
            ) from e

    def _extract_message_history_from_spans(
        self, trace_id: str
    ) -> List[Dict[str, str]]:
        """
        Extract message history from spans in the span store for training purposes.

        This method processes trace spans to reconstruct the conversation flow,
        extracting messages in chronological order from LLM, user, and tool spans.

        Args:
            trace_id: The trace ID (32-char hex string) to extract message history from

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
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

                # Add assistant response from span output
                output = span_attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
                if output is not None:
                    content = str(output)
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "messages" in parsed:
                            # Extract the actual assistant message content
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
        agent_function: Callable[[Any], Any],
        scorers: List[Union[ExampleAPIScorerConfig, ExampleScorer]],
        prompts: List[Any],
        num_prompts_per_step: Optional[int] = None,
        num_generations_per_prompt: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        """
        Generate rollouts and compute rewards using the current model snapshot.
        Each sample contains multiple generations for reinforcement learning optimization.

        Args:
            agent_function: Function/agent to call for generating responses
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training
            num_prompts_per_step: Number of prompts to use per step (defaults to config value, limited by prompts list length)
            num_generations_per_prompt: Generations per prompt (defaults to config value)
            concurrency: Concurrency limit (defaults to config value)

        Returns:
            List of dataset rows containing samples with messages and evaluations
        """
        num_prompts_per_step = min(
            num_prompts_per_step or self.config.num_prompts_per_step, len(prompts)
        )
        num_generations_per_prompt = (
            num_generations_per_prompt or self.config.num_generations_per_prompt
        )
        concurrency = concurrency or self.config.concurrency

        semaphore = asyncio.Semaphore(concurrency)

        @self.tracer.observe(span_type="function")
        async def generate_single_response(prompt_id, generation_id):
            async with semaphore:
                prompt_input = prompts[prompt_id]
                response_data = await agent_function(**prompt_input)
                messages = response_data.get("messages", [])

                current_span = self.tracer.get_current_span()
                trace_id = None
                if current_span and current_span.is_recording():
                    # Convert trace_id to hex string per OTEL spec
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
                    pass

                finally:
                    if trace_id is not None:
                        self.span_store.clear_trace(trace_id)

                example = Example(
                    input=prompt_input,
                    messages=messages,
                    actual_output=response_data,
                )

                scoring_results = self.judgment_client.run_evaluation(
                    examples=[example],
                    scorers=scorers,
                    project_name=self.project_name,
                    eval_run_name=f"training_step_{self.trainable_model.current_step}_prompt_{prompt_id}_gen_{generation_id}",
                )

                if scoring_results and scoring_results[0].scorers_data:
                    scores = [
                        scorer_data.score
                        for scorer_data in scoring_results[0].scorers_data
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

            for coro in asyncio.as_completed(coros):
                result = await coro
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

    async def run_reinforcement_learning(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[Union[ExampleAPIScorerConfig, ExampleScorer]],
        prompts: List[Any],
    ) -> ModelConfig:
        """
        Run the iterative reinforcement learning fine-tuning loop.

        This method performs multiple steps of reinforcement learning, where each step:
        1. Advances to the appropriate model snapshot
        2. Generates rollouts and computes rewards using scorers
        3. Trains a new model using reinforcement learning
        4. Waits for training completion

        Args:
            agent_function: Function/agent to call for generating responses
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training

        Returns:
            ModelConfig: Configuration of the trained model for inference and future training
        """

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
        scorers: List[Union[ExampleAPIScorerConfig, ExampleScorer]],
        prompts: List[Any],
    ) -> ModelConfig:
        """
        Start the reinforcement learning fine-tuning process.

        This is the main entry point for running the reinforcement learning training.

        Args:
            agent_function: Function/agent to call for generating responses.
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training

        Returns:
            ModelConfig: Configuration of the trained model for future loading
        """
        try:
            return await self.run_reinforcement_learning(
                agent_function, scorers, prompts
            )
        except JudgmentRuntimeError:
            # Re-raise JudgmentRuntimeError as-is
            raise
        except Exception as e:
            raise JudgmentRuntimeError(f"Training process failed: {str(e)}") from e
