from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from judgeval.v1.tracer.tracer import Tracer
    from judgeval.v1.trainers.trainable_model import TrainableModel
    from judgeval.v1.trainers.config import TrainerConfig, ModelConfig
    from judgeval.v1.scorers.base_scorer import BaseScorer


class BaseTrainer(ABC):
    __slots__ = ("config", "trainable_model", "tracer", "project_name")

    def __init__(
        self,
        config: TrainerConfig,
        trainable_model: TrainableModel,
        tracer: Tracer,
        project_name: Optional[str] = None,
    ):
        self.config = config
        self.trainable_model = trainable_model
        self.tracer = tracer
        self.project_name = project_name or "judgment_training"

    @abstractmethod
    async def generate_rollouts_and_rewards(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[BaseScorer],
        prompts: dict[int, dict[Any, Any]],
        num_prompts_per_step: Optional[int] = None,
        num_generations_per_prompt: Optional[int] = None,
        concurrency: Optional[int] = None,
    ) -> Any:
        pass

    @abstractmethod
    async def run_reinforcement_learning(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[BaseScorer],
        prompts: dict[int, dict[Any, Any]],
    ) -> "ModelConfig":
        pass

    @abstractmethod
    async def train(
        self,
        agent_function: Callable[[Any], Any],
        scorers: List[BaseScorer],
        prompts: dict[int, dict[Any, Any]],
    ) -> "ModelConfig":
        pass

    @abstractmethod
    def _extract_message_history_from_spans(
        self, trace_id: str
    ) -> List[Dict[str, str]]:
        pass
