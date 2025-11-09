from __future__ import annotations

from typing import TYPE_CHECKING

from judgeval.v1.internal.api import JudgmentSyncClient

if TYPE_CHECKING:
    from judgeval.trainer.config import TrainerConfig
    from judgeval.trainer.trainable_model import TrainableModel
    from judgeval.v1.tracer.tracer import Tracer


class TrainersFactory:
    __slots__ = "_client"

    def __init__(
        self,
        client: JudgmentSyncClient,
    ):
        self._client = client

    def fireworks(
        self,
        config: TrainerConfig,
        trainable_model: TrainableModel,
        tracer: Tracer,
        project_name: str | None = None,
    ):
        from judgeval.v1.trainers.fireworks_trainer import FireworksTrainer

        return FireworksTrainer(
            config=config,
            trainable_model=trainable_model,
            tracer=tracer,
            project_name=project_name,
            client=self._client,
        )
