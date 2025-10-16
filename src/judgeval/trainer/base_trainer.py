from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union, Dict, TYPE_CHECKING
from .config import TrainerConfig, ModelConfig
from judgeval.scorers import ExampleScorer, ExampleAPIScorerConfig

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from .trainable_model import TrainableModel


class BaseTrainer(ABC):
    """
    Abstract base class for training providers.

    This class defines the interface that all training provider implementations
    must follow. Each provider (Fireworks, Verifiers, etc.) will have its own
    concrete implementation of this interface.
    """

    def __init__(
        self,
        config: TrainerConfig,
        trainable_model: "TrainableModel",
        tracer: "Tracer",
        project_name: Optional[str] = None,
    ):
        """
        Initialize the base trainer.

        Args:
            config: TrainerConfig instance with training parameters
            trainable_model: TrainableModel instance to use for training
            tracer: Tracer for observability
            project_name: Project name for organizing training runs
        """
        self.config = config
        self.trainable_model = trainable_model
        self.tracer = tracer
        self.project_name = project_name or "judgment_training"

    @abstractmethod
    async def generate_rollouts_and_rewards(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[Union[ExampleAPIScorerConfig, ExampleScorer]],
        prompts: List[Any],
        num_prompts_per_step: Optional[int] = None,
        num_generations_per_prompt: Optional[int] = None,
        concurrency: Optional[int] = None,
    ) -> Any:
        """
        Generate rollouts and compute rewards using the current model snapshot.

        Args:
            agent_function: Function/agent to call for generating responses
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training
            num_prompts_per_step: Number of prompts to use per step
            num_generations_per_prompt: Generations per prompt
            concurrency: Concurrency limit

        Returns:
            Provider-specific dataset format for training
        """
        pass

    @abstractmethod
    async def run_reinforcement_learning(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[Union[ExampleAPIScorerConfig, ExampleScorer]],
        prompts: List[Any],
    ) -> ModelConfig:
        """
        Run the iterative reinforcement learning fine-tuning loop.

        Args:
            agent_function: Function/agent to call for generating responses
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training

        Returns:
            ModelConfig: Configuration of the trained model
        """
        pass

    @abstractmethod
    async def train(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[Union[ExampleAPIScorerConfig, ExampleScorer]],
        prompts: List[Any],
    ) -> ModelConfig:
        """
        Start the reinforcement learning fine-tuning process.

        This is the main entry point for running the training.

        Args:
            agent_function: Function/agent to call for generating responses
            scorers: List of scorer objects to evaluate responses
            prompts: List of prompts to use for training

        Returns:
            ModelConfig: Configuration of the trained model
        """
        pass

    @abstractmethod
    def _extract_message_history_from_spans(self) -> List[Dict[str, str]]:
        """
        Extract message history from spans for training purposes.

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        pass
