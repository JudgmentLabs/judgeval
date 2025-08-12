import asyncio
import time
from typing import Optional
from fireworks import LLM, Dataset
from .config import TrainerConfig
from judgeval.tracer import Tracer, wrap


class TrainableModel:
    """
    A wrapper class for managing model snapshots during training.

    This class automatically handles model snapshot creation and management
    during the GRPO (Generative Reinforcement Learning from Policy Optimization) process,
    abstracting away manual snapshot management from users.
    """

    def __init__(self, config: TrainerConfig):
        """
        Initialize the TrainableModel.

        Args:
            config: TrainerConfig instance with model configuration
        """
        self.config = config
        self.current_step = 0
        self._current_model = None

        # Initialize base model
        self._base_model = self._create_base_model()
        self._current_model = self._base_model

    def _create_base_model(self):
        """Create and configure the base model."""
        base_model = LLM(
            model=self.config.base_model_name,
            deployment_type="on-demand",
            id=self.config.deployment_id,
            enable_addons=self.config.enable_addons,
        )
        base_model.apply()
        return base_model

    def get_current_model(self):
        """Get the current model snapshot for generation."""
        return self._current_model

    @property
    def chat(self):
        """OpenAI-compatible chat interface."""
        return self._current_model.chat

    @property
    def completions(self):
        """OpenAI-compatible completions interface."""
        return self._current_model.completions

    def advance_to_next_step(self, step: int):
        """
        Advance to the next training step and update the current model snapshot.

        Args:
            step: The current training step number
        """
        self.current_step = step

        if step == 0:
            # Use base model for first step
            self._current_model = self._base_model
        else:
            # Create new model snapshot from previous training step
            model_name = (
                f"accounts/{self.config.user_id}/models/{self.config.model_id}-v{step}"
            )
            self._current_model = LLM(
                model=model_name,
                deployment_type="on-demand-lora",
                base_id=self.config.deployment_id,
            )
            # Ensure deployment is ready
            self._current_model.apply()

    def perform_reinforcement_step(self, dataset, step: int):
        """
        Perform a reinforcement learning step using the current model.

        Args:
            dataset: Training dataset for the reinforcement step
            step: Current step number for output model naming

        Returns:
            Training job object
        """
        model_name = f"{self.config.model_id}-v{step + 1}"
        return self._current_model.reinforcement_step(
            dataset=dataset,
            output_model=model_name,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            accelerator_count=self.config.accelerator_count,
            accelerator_type=self.config.accelerator_type,
        )


class JudgmentTrainer:
    """
    A reinforcement learning trainer for judgment models using Fireworks AI.

    This class handles the iterative training process where models are improved
    through reinforcement learning steps based on generated rollouts and rewards.
    """

    def __init__(
        self, config: Optional[TrainerConfig] = None, tracer: Optional[Tracer] = None
    ):
        """
        Initialize the JudgmentTrainer.

        Args:
            config: TrainerConfig instance with training parameters. If None, uses default config.
            tracer: Optional tracer for observability
        """
        self.config = config or TrainerConfig()
        self.tracer = tracer

        # Initialize trainable model wrapper
        self.trainable_model = wrap(TrainableModel(self.config))

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
        num_prompts: Optional[int] = None,
        num_generations_per_prompt: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        """
        Generate rollouts and compute rewards using the current model snapshot.
        Each sample contains multiple generations for Policy Optimization.

        Args:
            num_prompts: Number of prompts to use (defaults to config value)
            num_generations_per_prompt: Generations per prompt (defaults to config value)
            concurrency: Concurrency limit (defaults to config value)

        Returns:
            List of dataset rows containing samples with messages and evaluations
        """
        num_prompts = num_prompts or self.config.num_prompts
        num_generations_per_prompt = (
            num_generations_per_prompt or self.config.num_generations_per_prompt
        )
        concurrency = concurrency or self.config.concurrency

        semaphore = asyncio.Semaphore(concurrency)

        @self.tracer.observe(span_type="function")
        async def generate_single_response(prompt_id, generation_id):
            """Generate a single response for a given prompt."""
            async with semaphore:
                # Define the conversation turns
                user_prompts = [
                    f"What is {prompt_id} + {prompt_id}?",
                    "Now multiply that result by 2.",
                    "Now add 10 to the result.",
                ]

                messages = []
                responses = []

                # Loop through each turn
                for turn, user_prompt in enumerate(user_prompts):
                    # Add user message
                    messages.append({"role": "user", "content": user_prompt})

                    # Get model response
                    response = await self.trainable_model.chat.completions.acreate(
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        n=1,
                    )

                    assistant_response = response.choices[0].message.content
                    responses.append(assistant_response)

                    # Add assistant response to conversation
                    messages.append(
                        {"role": "assistant", "content": assistant_response}
                    )

                # Compute reward based on both responses
                first_correct = str(prompt_id + prompt_id) in responses[0]
                expected_final = (prompt_id + prompt_id) * 2
                second_correct = str(expected_final) in responses[1]
                third_correct = str(expected_final + 10) in responses[2]

                if first_correct and second_correct and third_correct:
                    reward = 1.0  # Both answers correct
                elif first_correct or second_correct or third_correct:
                    reward = 0.5  # Partial credit
                else:
                    reward = 0.0  # Both incorrect

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

    async def run_reinforcement_learning(self):
        """
        Run the iterative reinforcement learning loop.

        This method performs multiple steps of reinforcement learning, where each step:
        1. Advances to the appropriate model snapshot
        2. Generates rollouts and computes rewards
        3. Trains a new model using reinforcement learning
        4. Waits for training completion
        """

        print("Starting reinforcement learning")
        for step in range(self.config.num_steps):
            print(
                f"Starting reinforcement learning step {step + 1}/{self.config.num_steps}"
            )

            # Advance trainable model to the current step
            self.trainable_model.advance_to_next_step(step)

            # Generate rollouts and rewards using current model snapshot
            dataset_rows = await self.generate_rollouts_and_rewards()

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

    def train(self):
        """
        Start the training process.

        This is the main entry point for running the reinforcement learning training.
        """
        asyncio.run(self.run_reinforcement_learning())
