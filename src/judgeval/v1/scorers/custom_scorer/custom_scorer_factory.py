from __future__ import annotations

from typing import Optional

from judgeval.v1.scorers.custom_scorer.custom_scorer import CustomScorer


class CustomScorerFactory:
    __slots__ = ()

    def get(self, name: str, class_name: Optional[str] = None) -> CustomScorer:
        return CustomScorer(
            name=name,
            class_name=class_name or name,
            server_hosted=True,
        )

    def upload(
        self,
        scorer_file_path: str,
        requirements_file_path: Optional[str] = None,
        unique_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> bool:
        raise NotImplementedError("CustomScorer upload not implemented in v1")
