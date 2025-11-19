import time
from fireworks import LLM  # type: ignore[import-not-found,import-untyped]
from .config import TrainerConfig, ModelConfig
from typing import Optional, Dict, Any, Callable
from .console import _model_spinner_progress, _print_model_progress
from judgeval.exceptions import JudgmentRuntimeError


class TrainableModel:
    """
    A wrapper class for managing model snapshots during training.

    This class automatically handles model snapshot creation and management
    during the RFT (Reinforcement Fine-Tuning) process,
    abstracting away manual snapshot management from users.
    """

    config: TrainerConfig
    current_step: int
    _current_model: LLM
    _tracer_wrapper_func: Optional[Callable]
    _base_model: LLM

    def __init__(self, config: TrainerConfig):
        """
        Initialize the TrainableModel.

        Args:
            config: TrainerConfig instance with model configuration
        """
        try:
            self.config = config
            self.current_step = 0
            self._tracer_wrapper_func = None

            self._base_model = self._create_base_model()
            self._current_model = self._base_model
        except Exception as e:
            raise JudgmentRuntimeError(
                f"Failed to initialize TrainableModel: {str(e)}"
            ) from e

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

        instance = cls(trainer_config)
        instance.current_step = model_config.current_step

        if model_config.is_trained and model_config.current_model_name:
            instance._load_trained_model(model_config.current_model_name)

        return instance

    def _create_base_model(self):
        """Create and configure the base model."""
        try:
            with _model_spinner_progress(
                "Creating and deploying base model..."
            ) as update_progress:
                update_progress("Creating base model instance...")
                base_model = LLM(
                    model=self.config.base_model_name,
                    deployment_type="on-demand",
                    id=self.config.deployment_id,
                    enable_addons=self.config.enable_addons,
                )
                update_progress("Applying deployment configuration...")
                base_model.apply()
            _print_model_progress("Base model deployment ready")
            return base_model
        except Exception as e:
            raise JudgmentRuntimeError(
                f"Failed to create and deploy base model '{self.config.base_model_name}': {str(e)}"
            ) from e

    def _load_trained_model(self, model_name: str):
        """Load a trained model by name."""
        try:
            with _model_spinner_progress(
                f"Loading and deploying trained model: {model_name}"
            ) as update_progress:
                update_progress("Creating trained model instance...")
                self._current_model = LLM(
                    model=model_name,
                    deployment_type="on-demand-lora",
                    base_id=self.config.deployment_id,
                )
                update_progress("Applying deployment configuration...")
                self._current_model.apply()
            _print_model_progress("Trained model deployment ready")

            if self._tracer_wrapper_func:
                self._tracer_wrapper_func(self._current_model)
        except Exception as e:
            raise JudgmentRuntimeError(
                f"Failed to load and deploy trained model '{model_name}': {str(e)}"
            ) from e

    def get_current_model(self):
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
        try:
            self.current_step = step

            if step == 0:
                self._current_model = self._base_model
            else:
                model_name = f"accounts/{self.config.user_id}/models/{self.config.model_id}-v{step}"
                with _model_spinner_progress(
                    f"Creating and deploying model snapshot: {model_name}"
                ) as update_progress:
                    update_progress("Creating model snapshot instance...")
                    self._current_model = LLM(
                        model=model_name,
                        deployment_type="on-demand-lora",
                        base_id=self.config.deployment_id,
                    )
                    update_progress("Applying deployment configuration...")
                    self._current_model.apply()
                _print_model_progress("Model snapshot deployment ready")

                if self._tracer_wrapper_func:
                    self._tracer_wrapper_func(self._current_model)
        except Exception as e:
            raise JudgmentRuntimeError(
                f"Failed to advance to training step {step}: {str(e)}"
            ) from e

    def perform_reinforcement_step(
        self, dataset, step: int, max_retries: int = 3, initial_backoff: float = 1.0
    ):
        """
        Perform a reinforcement learning step using the current model.

        Args:
            dataset: Training dataset for the reinforcement step
            step: Current step number for output model naming
            max_retries: Maximum number of retry attempts (default: 3)
            initial_backoff: Initial backoff time in seconds for exponential backoff (default: 1.0)

        Returns:
            Training job object
        """
        model_name = f"{self.config.model_id}-v{step + 1}"

        for attempt in range(max_retries):
            try:
                return self._current_model.reinforcement_step(
                    dataset=dataset,
                    output_model=model_name,
                    epochs=self.config.epochs,
                    learning_rate=self.config.learning_rate,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff_time = initial_backoff * (2**attempt)
                    time.sleep(backoff_time)
                else:
                    raise JudgmentRuntimeError(
                        f"Failed to start reinforcement learning step {step + 1} after {max_retries} attempts: {str(e)}"
                    ) from e

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

    def _register_tracer_wrapper(self, wrapper_func: Callable):
        """
        Register a tracer wrapper function to be reapplied when models change.

        This is called internally by the tracer's wrap() function to ensure
        that new model instances created during training are automatically wrapped.

        Args:
            wrapper_func: Function that wraps a model instance with tracing
        """
        self._tracer_wrapper_func = wrapper_func
