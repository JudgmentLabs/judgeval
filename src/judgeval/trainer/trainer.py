from typing import Optional
from .config import TrainerConfig
from .base_trainer import BaseTrainer
from .fireworks_trainer import FireworksTrainer
from .trainable_model import TrainableModel
from judgeval.tracer import Tracer
from judgeval.exceptions import JudgmentRuntimeError


def JudgmentTrainer(
    config: TrainerConfig,
    trainable_model: TrainableModel,
    tracer: Tracer,
    project_name: Optional[str] = None,
) -> BaseTrainer:
    """
    Factory function for creating reinforcement learning trainers.

    This factory creates and returns provider-specific trainer implementations
    (FireworksTrainer, VerifiersTrainer, etc.) based on the configured RFT provider.

    The factory pattern allows for easy extension to support multiple training
    providers without changing the client-facing API.

    Example:
        config = TrainerConfig(
            deployment_id="my-deployment",
            user_id="my-user",
            model_id="my-model",
            rft_provider="fireworks"  # or "verifiers" in the future
        )

        # User creates and configures the trainable model
        trainable_model = TrainableModel(config)
        tracer = Tracer()

        # JudgmentTrainer automatically creates the appropriate provider-specific trainer
        trainer = JudgmentTrainer(config, trainable_model, tracer)

        # The returned trainer implements the BaseTrainer interface
        model_config = await trainer.train(agent_function, scorers, prompts)

        Args:
        config: TrainerConfig instance with training parameters including rft_provider
        trainable_model: Provider-specific trainable model instance (e.g., TrainableModel for Fireworks)
        tracer: Tracer for observability
        project_name: Project name for organizing training runs and evaluations

        Returns:
        Provider-specific trainer instance (FireworksTrainer, etc.) that implements
        the BaseTrainer interface

    Raises:
        JudgmentRuntimeError: If the specified provider is not supported
    """
    provider = config.rft_provider.lower()

    if provider == "fireworks":
        return FireworksTrainer(config, trainable_model, tracer, project_name)
    elif provider == "verifiers":
        # Placeholder for future implementation
        raise JudgmentRuntimeError(
            "Verifiers provider is not yet implemented. "
            "Currently supported providers: 'fireworks'"
        )
    else:
        raise JudgmentRuntimeError(
            f"Unsupported RFT provider: '{config.rft_provider}'. "
            f"Currently supported providers: 'fireworks'"
        )
