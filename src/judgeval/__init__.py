# Import key components that should be publicly accessible
from judgeval.clients import client, together_client
from judgeval.judgment_client import JudgmentClient
from judgeval.version_check import check_latest_version
from judgeval.local_eval_queue import LocalEvaluationQueue

check_latest_version()

__all__ = [
    # Clients
    "client",
    "together_client",
    "JudgmentClient",
    "LocalEvaluationQueue",
]
