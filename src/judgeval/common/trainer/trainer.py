import asyncio
import time
from typing import Optional, Callable, Any, List
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
        agent_function: Callable[[Any], Any],
        reward_function: Callable[[Any, Any], float],
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
            reward_function: Function to compute reward given prompt and response
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
                response_data = await agent_function(
                    model=self.trainable_model,
                    prompt_input=prompt_input,
                    config=self.config,
                )

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

                # Compute reward using provided reward function
                reward = reward_function(prompt_input, response_data)

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
        reward_function: Callable[[Any, Any], float],
        prompts: List[Any],
    ):
        """
        Run the iterative reinforcement learning loop.

        This method performs multiple steps of reinforcement learning, where each step:
        1. Advances to the appropriate model snapshot
        2. Generates rollouts and computes rewards
        3. Trains a new model using reinforcement learning
        4. Waits for training completion

        Args:
            agent_function: Function/agent to call for generating responses
            reward_function: Function to compute reward given prompt and response
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
                agent_function, reward_function, prompts
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
        reward_function: Callable[[Any, Any], float],
        prompts: List[Any],
    ):
        """
        Start the training process.

        This is the main entry point for running the reinforcement learning training.

        Args:
            agent_function: Function/agent to call for generating responses.
                           Should accept (model, prompt_input, config) and return response data
            reward_function: Function to compute reward given (prompt_input, response_data)
            prompts: List of prompts to use for training
        """
        asyncio.run(
            self.run_reinforcement_learning(agent_function, reward_function, prompts)
        )
