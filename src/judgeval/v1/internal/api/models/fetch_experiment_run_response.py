from __future__ import annotations

from typing import TypedDict, Optional, List
from typing_extensions import NotRequired

from .experiment_run_item import ExperimentRunItem


class FetchExperimentRunResponse(TypedDict):
    results: NotRequired[Optional[List[ExperimentRunItem]]]
    ui_results_url: NotRequired[Optional[str]]
