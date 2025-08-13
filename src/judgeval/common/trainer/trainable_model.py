from fireworks import LLM
from .config import TrainerConfig, ModelConfig
from typing import Optional, Dict, Any


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

    @classmethod
    def from_model_config(cls, model_config: ModelConfig) -> "TrainableModel":
        """
        Create a TrainableModel from a saved ModelConfig.

        Args:
            model_config: ModelConfig instance with saved model state

        Returns:
            TrainableModel instance configured to use the saved model
        """
        # Create a TrainerConfig from the ModelConfig
        trainer_config = TrainerConfig(
            base_model_name=model_config.base_model_name,
            deployment_id=model_config.deployment_id,
            user_id=model_config.user_id,
            model_id=model_config.model_id,
            enable_addons=model_config.enable_addons,
        )

        # Create instance
        instance = cls(trainer_config)

        # Restore the training state
        instance.current_step = model_config.current_step

        # If there's a trained model, load it
        if model_config.is_trained and model_config.current_model_name:
            instance._load_trained_model(model_config.current_model_name)

        return instance

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

    def _load_trained_model(self, model_name: str):
        """Load a trained model by name."""
        self._current_model = LLM(
            model=model_name,
            deployment_type="on-demand-lora",
            base_id=self.config.deployment_id,
        )
        self._current_model.apply()

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

    def get_model_config(
        self, training_params: Optional[Dict[str, Any]] = None
    ) -> ModelConfig:
        """
        Get the current model configuration for persistence.

        Args:
            training_params: Optional training parameters to include in config

        Returns:
            ModelConfig instance with current model state
        """
        # Determine current model name
        current_model_name = None
        is_trained = False

        if self.current_step > 0:
            current_model_name = f"accounts/{self.config.user_id}/models/{self.config.model_id}-v{self.current_step}"
            is_trained = True

        return ModelConfig(
            base_model_name=self.config.base_model_name,
            deployment_id=self.config.deployment_id,
            user_id=self.config.user_id,
            model_id=self.config.model_id,
            enable_addons=self.config.enable_addons,
            current_step=self.current_step,
            total_steps=self.config.num_steps,
            current_model_name=current_model_name,
            is_trained=is_trained,
            training_params=training_params,
        )

    def save_model_config(
        self, filepath: str, training_params: Optional[Dict[str, Any]] = None
    ):
        """
        Save the current model configuration to a file.

        Args:
            filepath: Path to save the configuration file
            training_params: Optional training parameters to include in config
        """
        model_config = self.get_model_config(training_params)
        model_config.save_to_file(filepath)
