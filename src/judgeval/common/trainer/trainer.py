import asyncio
import time
from typing import Optional, Callable, Any, List, Union
from fireworks import Dataset
from .config import TrainerConfig, ModelConfig
from .trainable_model import TrainableModel
from judgeval.tracer import Tracer
from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.data import Example
from judgeval.utils.async_utils import safe_run_async
from .console import _spinner_progress, _print_progress, _print_progress_update


class JudgmentTrainer:
    """
    A reinforcement learning trainer for judgment models using Fireworks AI.

    This class handles the iterative training process where models are improved
    through reinforcement learning steps based on generated rollouts and rewards.
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        trainable_model: Optional[TrainableModel] = None,
        tracer: Optional[Tracer] = None,
        project_name: Optional[str] = None,
    ):
        """
        Initialize the JudgmentTrainer.

        Args:
            config: TrainerConfig instance with training parameters. If None, uses default config.
            tracer: Optional tracer for observability
            trainable_model: Optional trainable model instance
            project_name: Project name for organizing training runs and evaluations
        """
        self.config = config or TrainerConfig()
        self.tracer = tracer
        self.project_name = project_name or "judgment_training"

        # Initialize trainable model wrapper
        if trainable_model is None:
            self.trainable_model = TrainableModel(self.config)
        else:
            self.trainable_model = trainable_model

        # Initialize judgment client for evaluation
        self.judgment_client = JudgmentClient()

    async def generate_rollouts_and_rewards(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        prompts: List[Any],
        num_prompts: Optional[int] = None,
        num_generations_per_prompt: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        """
        Generate rollouts and compute rewards using the current model snapshot.
        Each sample contains multiple generations for Policy Optimization.

        Args:
            agent_function: Function/agent to call for generating responses
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training
            num_prompts: Number of prompts to use (defaults to config value, limited by prompts list length)
            num_generations_per_prompt: Generations per prompt (defaults to config value)
            concurrency: Concurrency limit (defaults to config value)

        Returns:
            List of dataset rows containing samples with messages and evaluations
        """
        num_prompts = min(num_prompts or self.config.num_prompts, len(prompts))
        num_generations_per_prompt = (
            num_generations_per_prompt or self.config.num_generations_per_prompt
        )
        concurrency = concurrency or self.config.concurrency

        semaphore = asyncio.Semaphore(concurrency)

        @self.tracer.observe(span_type="function")
        async def generate_single_response(prompt_id, generation_id):
            """Generate a single response for a given prompt."""
            async with semaphore:
                # Get prompt from the provided list
                prompt_input = prompts[prompt_id]

                # Call the agent function with the current model and prompt
                response_data = await agent_function(**prompt_input)

                # Extract messages from response_data or trace
                messages = response_data.get("messages", [])

                # Extract the actual conversation from the trace if available
                try:
                    traced_messages = self.tracer.get_current_message_history()
                    if traced_messages:
                        messages = traced_messages
                except Exception as e:
                    # Fallback to response_data messages if trace extraction fails, but log the error
                    print(f"Warning: Failed to get message history from trace: {e}")
                    pass

                # Create an Example object from the response data for evaluation
                # Include prompt_input, messages, and response_data as requested
                example = Example(
                    input=prompt_input, messages=messages, actual_output=response_data
                )

                # Use run_evaluation to compute reward using scorer objects
                scoring_results = self.judgment_client.run_evaluation(
                    examples=[example],
                    scorers=scorers,
                    project_name=self.project_name,
                    eval_run_name=f"training_step_{self.trainable_model.current_step}_prompt_{prompt_id}_gen_{generation_id}",
                )

                # Extract reward from scoring results
                # Take the average score across all scorers as the reward
                if scoring_results and scoring_results[0].scorers_data:
                    reward = sum(
                        scorer_data.score
                        for scorer_data in scoring_results[0].scorers_data
                    ) / len(scoring_results[0].scorers_data)
                else:
                    reward = 0.0

            return {
                "prompt_id": prompt_id,
                "generation_id": generation_id,
                "messages": messages,
                "evals": {"score": reward},
            }

        # Create all generation tasks concurrently
        coros = []
        for prompt_id in range(num_prompts):
            for generation_id in range(num_generations_per_prompt):
                coro = generate_single_response(prompt_id, generation_id)
                coros.append(coro)

        # Execute all generations concurrently
        with _spinner_progress(f"Generating {len(coros)} rollouts..."):
            num_completed = 0
            results = []

            for coro in asyncio.as_completed(coros):
                result = await coro
                results.append(result)
                num_completed += 1
                # Don't print intermediate progress during spinner operation

        _print_progress(f"Generated {len(results)} rollouts successfully")

        # Group results by prompt_id to create dataset rows
        dataset_rows = []
        for prompt_id in range(num_prompts):
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
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        prompts: List[Any],
    ) -> ModelConfig:
        """
        Run the iterative reinforcement learning loop.

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
            ModelConfig: Configuration of the trained model for future loading
        """

        _print_progress("Starting reinforcement learning training")

        # Store training parameters for the model config
        training_params = {
            "num_steps": self.config.num_steps,
            "num_prompts": self.config.num_prompts,
            "num_generations_per_prompt": self.config.num_generations_per_prompt,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "accelerator_count": self.config.accelerator_count,
            "accelerator_type": self.config.accelerator_type,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Start from the current step of the model (useful for resuming training)
        start_step = self.trainable_model.current_step

        for step in range(start_step, self.config.num_steps):
            step_num = step + 1
            _print_progress(
                f"Starting training step {step_num}", step_num, self.config.num_steps
            )

            # Advance trainable model to the current step
            if step > 0:
                self.trainable_model.advance_to_next_step(step)
            else:
                self.trainable_model.advance_to_next_step(step)

            # Generate rollouts and rewards using current model snapshot
            dataset_rows = await self.generate_rollouts_and_rewards(
                agent_function, scorers, prompts
            )

            # Create dataset from dataset rows
            with _spinner_progress(
                "Preparing training dataset", step_num, self.config.num_steps
            ):
                dataset = Dataset.from_list(dataset_rows)
                dataset.sync()

            # Perform reinforcement learning step using trainable model
            _print_progress(
                "Starting reinforcement training", step_num, self.config.num_steps
            )
            job = self.trainable_model.perform_reinforcement_step(dataset, step)

            # Wait for training completion with better progress indicators
            last_state = None
            with _spinner_progress(
                "Training job in progress", step_num, self.config.num_steps
            ):
                while not job.is_completed:
                    job.raise_if_bad_state()
                    current_state = job.state

                    # Only print state changes to avoid spam
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

                    time.sleep(10)
                    job = job.get()
                    if job is None:
                        raise Exception("Job was deleted while waiting for completion")

            _print_progress(
                f"Training completed! New model: {job.output_model}",
                step_num,
                self.config.num_steps,
            )

            # Clean up dataset
            dataset.delete()

        _print_progress("All training steps completed!")

        # Update the model to the final step
        with _spinner_progress("Deploying final trained model"):
            self.trainable_model.advance_to_next_step(self.config.num_steps)

        # Return the final model configuration
        return self.trainable_model.get_model_config(training_params)

    def train(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        prompts: List[Any],
    ) -> ModelConfig:
        """
        Start the training process.

        This is the main entry point for running the reinforcement learning training.

        Args:
            agent_function: Function/agent to call for generating responses.
                           Should accept (model, prompt_input, config) and return response data
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training

        Returns:
            ModelConfig: Configuration of the trained model for future loading
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, we're in an async context - use nest_asyncio or thread approach
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(
                self.run_reinforcement_learning(agent_function, scorers, prompts)
            )
        except RuntimeError:
            # No running loop, safe to use safe_run_async
            return safe_run_async(
                self.run_reinforcement_learning(agent_function, scorers, prompts)
            )
        except ImportError:
            # nest_asyncio not available, fall back to thread approach
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        self.run_reinforcement_learning(
                            agent_function, scorers, prompts
                        )
                    )
                )
                return future.result()

    async def train_async(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        prompts: List[Any],
    ) -> ModelConfig:
        """
        Async version of the training process.

        This is the async entry point for running the reinforcement learning training.
        Use this method if you're already in an async context.

        Args:
            agent_function: Function/agent to call for generating responses.
                           Should accept (model, prompt_input, config) and return response data
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training

        Returns:
            ModelConfig: Configuration of the trained model for future loading
        """
        return await self.run_reinforcement_learning(agent_function, scorers, prompts)
