from dataclasses import dataclass
from typing import Optional, Dict, Any
import json


@dataclass
class TrainerConfig:
    """Configuration class for JudgmentTrainer parameters."""

    deployment_id: str
    user_id: str
    model_id: str
    base_model_name: str = "qwen2p5-7b-instruct"
    rft_provider: str = "fireworks"
    num_steps: int = 5
    num_generations_per_prompt: int = (
        5  # Number of rollouts/generations per input prompt
    )
    num_prompts_per_step: int = 4  # Number of input prompts to sample per training step
    concurrency: int = 100
    epochs: int = 1
    learning_rate: float = 1e-5
    accelerator_count: int = 1
    accelerator_type: str = "NVIDIA_A100_80GB"
    temperature: float = 1.5
    max_tokens: int = 50
    enable_addons: bool = True


@dataclass
class ModelConfig:
    """
    Configuration class for storing and loading trained model state.

    This class enables persistence of trained models so they can be loaded
    and used later without retraining.

    Example usage:
        trainer = JudgmentTrainer(config)
        model_config = trainer.train(agent_function, scorers, prompts)

        # Save the trained model configuration
        model_config.save_to_file("my_trained_model.json")

        # Later, load and use the trained model
        loaded_config = ModelConfig.load_from_file("my_trained_model.json")
        trained_model = TrainableModel.from_model_config(loaded_config)

        # Use the trained model for inference
        response = trained_model.chat.completions.create(
            model="current",  # Uses the loaded trained model
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """

    # Base model configuration
    base_model_name: str
    deployment_id: str
    user_id: str
    model_id: str
    enable_addons: bool

    # Training state
    current_step: int
    total_steps: int

    # Current model information
    current_model_name: Optional[str] = None
    is_trained: bool = False

    # Training parameters used (for reference)
    training_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary for serialization."""
        return {
            "base_model_name": self.base_model_name,
            "deployment_id": self.deployment_id,
            "user_id": self.user_id,
            "model_id": self.model_id,
            "enable_addons": self.enable_addons,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_model_name": self.current_model_name,
            "is_trained": self.is_trained,
            "training_params": self.training_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(
            base_model_name=data.get("base_model_name", "qwen2p5-7b-instruct"),
            deployment_id=data.get("deployment_id", "my-base-deployment"),
            user_id=data.get("user_id", ""),
            model_id=data.get("model_id", ""),
            enable_addons=data.get("enable_addons", True),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            current_model_name=data.get("current_model_name"),
            is_trained=data.get("is_trained", False),
            training_params=data.get("training_params"),
        )

    def to_json(self) -> str:
        """Convert ModelConfig to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelConfig":
        """Create ModelConfig from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_to_file(self, filepath: str):
        """Save ModelConfig to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, filepath: str) -> "ModelConfig":
        """Load ModelConfig from a JSON file."""
        with open(filepath, "r") as f:
            json_str = f.read()
        return cls.from_json(json_str)
