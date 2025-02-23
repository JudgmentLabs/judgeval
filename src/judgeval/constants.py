"""
Constant variables used throughout source code
"""

from enum import Enum
import litellm
import os

class APIScorer(str, Enum):  
    """
    Collection of proprietary scorers implemented by Judgment.

    These are ready-made evaluation scorers that can be used to evaluate
    Examples via the Judgment API.
    """
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    ANSWER_CORRECTNESS = "answer_correctness"
    HALLUCINATION = "hallucination"
    SUMMARIZATION = "summarization"
    CONTEXTUAL_RECALL = "contextual_recall"
    CONTEXTUAL_RELEVANCY = "contextual_relevancy"
    CONTEXTUAL_PRECISION = "contextual_precision"
    TOOL_CORRECTNESS = "tool_correctness"
    JSON_CORRECTNESS = "json_correctness"

    @classmethod
    def _missing_(cls, value):
        # Handle case-insensitive lookup
        for member in cls:
            if member.value == value.lower():
                return member

ROOT_API = os.getenv("JUDGMENT_API_URL", "https://api.judgmentlabs.ai")
# API URLs
JUDGMENT_EVAL_API_URL = f"{ROOT_API}/evaluate/"
JUDGMENT_DATASETS_PUSH_API_URL = f"{ROOT_API}/datasets/push/"
JUDGMENT_DATASETS_PULL_API_URL = f"{ROOT_API}/datasets/pull/"
JUDGMENT_DATASETS_PULL_ALL_API_URL = f"{ROOT_API}/datasets/get_all_stats/"
JUDGMENT_EVAL_LOG_API_URL = f"{ROOT_API}/log_eval_results/"
JUDGMENT_EVAL_FETCH_API_URL = f"{ROOT_API}/fetch_eval_results/"
JUDGMENT_EVAL_DELETE_API_URL = f"{ROOT_API}/delete_eval_results_by_project_and_run_name/"
JUDGMENT_EVAL_DELETE_PROJECT_API_URL = f"{ROOT_API}/delete_eval_results_by_project/"
JUDGMENT_TRACES_FETCH_API_URL = f"{ROOT_API}/traces/fetch/"
JUDGMENT_TRACES_SAVE_API_URL = f"{ROOT_API}/traces/save/"
JUDGMENT_TRACES_DELETE_API_URL = f"{ROOT_API}/traces/delete/"

# RabbitMQ
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq-networklb-faa155df16ec9085.elb.us-west-1.amazonaws.com")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", 5672)
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "task_queue")

# Models
TOGETHER_SUPPORTED_MODELS = {
    "QWEN": "Qwen/Qwen2-72B-Instruct",
    "LLAMA3_70B_INSTRUCT_TURBO": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "LLAMA3_405B_INSTRUCT_TURBO": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "LLAMA3_8B_INSTRUCT_TURBO": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "MISTRAL_8x22B_INSTRUCT": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "MISTRAL_8x7B_INSTRUCT": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

JUDGMENT_SUPPORTED_MODELS = {"osiris-large", "osiris-mini"}

ACCEPTABLE_MODELS = set(litellm.model_list) | set(TOGETHER_SUPPORTED_MODELS.keys()) | JUDGMENT_SUPPORTED_MODELS

## System settings
MAX_WORKER_THREADS = 10
