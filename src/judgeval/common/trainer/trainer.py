import asyncio
import time
from typing import Optional, Callable, Any, List, Union
from fireworks import LLM, Dataset
from .config import TrainerConfig
from .trainable_model import TrainableModel
from judgeval.tracer import Tracer
from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.data import Example


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
        self.trainable_model = trainable_model

        # Initialize judgment client for evaluation
        self.judgment_client = JudgmentClient()

    def _initialize_base_model(self):
        """Initialize the base model with PEFT addon support."""

        base_model = LLM(
            model=self.config.base_model_name,
            deployment_type="on-demand",
            id=self.config.deployment_id,
            enable_addons=self.config.enable_addons,
        )

        # Apply deployment configuration to Fireworks
        base_model.apply()
        return base_model

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
                response_data = await agent_function(prompt_input)

                # Extract messages from response_data or trace
                messages = response_data.get("messages", [])

                # Extract the actual conversation from the trace if available
                try:
                    print("messages", messages)
                    traced_messages = self.tracer.get_current_message_history()
                    if traced_messages:
                        messages = traced_messages
                    print("traced_messages", traced_messages)
                except Exception:
                    # Fallback to response_data messages if trace extraction fails
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
        print(f"Starting {len(coros)} concurrent generations...")
        num_completed = 0
        results = []

        for coro in asyncio.as_completed(coros):
            result = await coro
            results.append(result)
            num_completed += 1
            if num_completed % 10 == 0:
                print(f"Completed {num_completed}/{len(coros)} generations")

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
    ):
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
        """

        print("Starting reinforcement learning")
        for step in range(self.config.num_steps):
            print(
                f"Starting reinforcement learning step {step + 1}/{self.config.num_steps}"
            )

            # Advance trainable model to the current step
            self.trainable_model.advance_to_next_step(step)

            # Generate rollouts and rewards using current model snapshot
            dataset_rows = await self.generate_rollouts_and_rewards(
                agent_function, scorers, prompts
            )

            # Create dataset from dataset rows
            dataset = Dataset.from_list(dataset_rows)
            dataset.sync()

            # Perform reinforcement learning step using trainable model
            job = self.trainable_model.perform_reinforcement_step(dataset, step)

            # Wait for training completion
            while not job.is_completed:
                job.raise_if_bad_state()
                print(f"Training state: {job.state}")
                time.sleep(10)
                job = job.get()
                if job is None:
                    raise Exception("Job was deleted while waiting for completion")

            print(f"Step {step + 1} completed! New model: {job.output_model}")

            # Clean up dataset
            dataset.delete()

        print("Reinforcement learning complete!")

    def train(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[Union[APIScorerConfig, BaseScorer]],
        prompts: List[Any],
    ):
        """
        Start the training process.

        This is the main entry point for running the reinforcement learning training.

        Args:
            agent_function: Function/agent to call for generating responses.
                           Should accept (model, prompt_input, config) and return response data
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training
        """
        asyncio.run(self.run_reinforcement_learning(agent_function, scorers, prompts))
