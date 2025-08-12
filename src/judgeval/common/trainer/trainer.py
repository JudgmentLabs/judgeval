import asyncio
import time
from typing import Optional
from fireworks import LLM, Dataset
from .config import TrainerConfig


class JudgmentTrainer:
    """
    A reinforcement learning trainer for judgment models using Fireworks AI.

    This class handles the iterative training process where models are improved
    through reinforcement learning steps based on generated rollouts and rewards.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        """
        Initialize the JudgmentTrainer.

        Args:
            config: TrainerConfig instance with training parameters. If None, uses default config.
        """
        self.config = config or TrainerConfig()

        # Initialize base model
        self.base_model = self._initialize_base_model()

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
        llm,
        num_prompts: Optional[int] = None,
        num_generations_per_prompt: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        """
        Generate rollouts and compute rewards for the given model using concurrent generation.
        Each sample contains multiple generations for Policy Optimization.

        Args:
            llm: The language model to use for generation
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

        async def generate_single_response(prompt_id, generation_id):
            """Generate a single response for a given prompt."""
            async with semaphore:
                messages = [
                    {"role": "user", "content": f"What is {prompt_id} + {prompt_id}?"}
                ]

                response = await llm.chat.completions.acreate(
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    n=1,  # Generate one response at a time
                )

                assistant_message = response.choices[0].message.content

                # Compute reward for this generation
                if str(prompt_id + prompt_id) in assistant_message:
                    reward = 1.0  # Correct answer
                else:
                    reward = 0.0  # Incorrect answer

                return {
                    "prompt_id": prompt_id,
                    "generation_id": generation_id,
                    "messages": messages
                    + [{"role": "assistant", "content": assistant_message}],
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
        1. Creates or loads a model snapshot
        2. Generates rollouts and computes rewards
        3. Trains a new model using reinforcement learning
        4. Waits for training completion
        """

        print("Starting reinforcement learning")
        for step in range(self.config.num_steps):
            print(
                f"Starting reinforcement learning step {step + 1}/{self.config.num_steps}"
            )

            model_snapshot = None
            # Create deployment for current model snapshot
            if step == 0:
                # Use base model for first step
                model_snapshot = self.base_model
            else:
                model_name = f"accounts/minhp/models/improved-model-v{step}"
                print("model_name", model_name)
                model_snapshot = LLM(
                    model=model_name,  # Use the LoRA model directly
                    deployment_type="on-demand-lora",
                    base_id=self.config.deployment_id,  # Use the same deployment ID
                )

            # Ensure deployment is ready
            model_snapshot.apply()

            # Generate rollouts and rewards
            dataset_rows = await self.generate_rollouts_and_rewards(model_snapshot)

            # Create dataset from dataset rows
            dataset = Dataset.from_list(dataset_rows)
            dataset.sync()

            # Perform reinforcement learning step
            job = model_snapshot.reinforcement_step(
                dataset=dataset,
                output_model=f"improved-model-v{step + 1}",
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                accelerator_count=self.config.accelerator_count,
                accelerator_type=self.config.accelerator_type,
            )

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
