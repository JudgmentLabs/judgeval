from fireworks import LLM
from .config import TrainerConfig


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
