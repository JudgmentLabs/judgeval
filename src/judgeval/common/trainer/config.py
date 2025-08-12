from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """Configuration class for JudgmentTrainer parameters."""

    base_model_name: str = "qwen2p5-7b-instruct"
    deployment_id: str = "my-base-deployment"
    user_id: str = "minhp"  # User ID for model naming
    model_id: str = "test-improved-model-multi-turn-v7"  # Base model ID for naming
    num_steps: int = 5
    num_prompts: int = (
        5  # Number of rollouts per input (was num_generations_per_prompt)
    )
    num_generations_per_prompt: int = (
        2  # Number of inputs to sample per step (was num_prompts)
    )
    concurrency: int = 100
    epochs: int = 1
    learning_rate: float = 1e-5
    accelerator_count: int = 1
    accelerator_type: str = "NVIDIA_A100_80GB"
    temperature: float = 1.5
    max_tokens: int = 50
    enable_addons: bool = True
