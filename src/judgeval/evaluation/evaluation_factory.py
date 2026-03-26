from __future__ import annotations

from typing import Optional

from judgeval.internal.api import JudgmentSyncClient
from judgeval.evaluation.evaluation import Evaluation


class EvaluationFactory:
    """Creates `Evaluation` instances for running batch scoring.

    Access this via `client.evaluation` -- you don't instantiate it directly.

    Examples:
        ```python
        evaluation = client.evaluation.create()
        results = evaluation.run(
            examples=examples,
            scorers=["faithfulness"],
            eval_run_name="nightly",
        )
        ```
    """

    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    def create(self) -> Evaluation:
        """Create a new `Evaluation` bound to the current project.

        Returns:
            An `Evaluation` instance ready for `.run()`.
        """
        return Evaluation(
            client=self._client,
            project_id=self._project_id,
            project_name=self._project_name,
        )
